import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import shapely.ops
import shapely.wkt
from dolphin._types import Filename, PathOrStr
from dolphin.io import S3Path, format_nc_filename, load_gdal, write_arr
from opera_utils import get_frame_bbox, group_by_burst
from osgeo import gdal
from pyproj import Transformer
from scipy import ndimage
from shapely.geometry import LinearRing, Polygon, box

EARTH_APPROX_CIRCUMFERENCE = 40075017.0
EARTH_RADIUS = EARTH_APPROX_CIRCUMFERENCE / (2 * np.pi)

MASK_S3_URL = "s3://opera-water-mask/v0.3/EPSG4326.vrt"
gdal.UseExceptions()
logger = logging.getLogger(__name__)

__all__ = ["create_water_mask"]


def margin_km_to_deg(margin_in_km):
    """Convert a margin value from kilometers to degrees."""
    km_to_deg_at_equator = 1000.0 / (EARTH_APPROX_CIRCUMFERENCE / 360.0)
    margin_in_deg = margin_in_km * km_to_deg_at_equator

    return margin_in_deg


def margin_km_to_longitude_deg(margin_in_km, lat=0):
    """Convert a margin value from kilometers to degrees as a function of latitude."""
    delta_lon = (
        180 * 1000 * margin_in_km / (np.pi * EARTH_RADIUS * np.cos(np.pi * lat / 180))
    )

    return delta_lon


def check_dateline(poly):
    """Split `poly` if it crosses the dateline.

    Parameters
    ----------
    poly : shapely.geometry.Polygon
        Input polygon.

    Returns
    -------
    polys : list of shapely.geometry.Polygon
        A list containing: the input polygon if it didn't cross the dateline, or
        two polygons otherwise (one on either side of the dateline).

    """
    x_min, _, x_max, _ = poly.bounds

    # Check dateline crossing
    if (x_max - x_min > 180.0) or (x_min <= 180.0 <= x_max):
        dateline = shapely.wkt.loads("LINESTRING( 180.0 -90.0, 180.0 90.0)")

        # build new polygon with all longitudes between 0 and 360
        x, y = poly.exterior.coords.xy
        new_x = (k + (k <= 0.0) * 360 for k in x)
        new_ring = LinearRing(zip(new_x, y, strict=True))

        # Split input polygon
        # (https://gis.stackexchange.com/questions/232771/splitting-polygon-by-linestring-in-geodjango_)
        merged_lines = shapely.ops.linemerge([dateline, new_ring])
        border_lines = shapely.ops.unary_union(merged_lines)
        decomp = shapely.ops.polygonize(border_lines)

        polys = list(decomp)

        for polygon_count in range(len(polys)):
            x, y = polys[polygon_count].exterior.coords.xy
            # if there are no longitude values above 180, continue
            if not any(k > 180 for k in x):
                continue

            # otherwise, wrap longitude values down by 360 degrees
            x_wrapped_minus_360 = np.asarray(x) - 360
            polys[polygon_count] = Polygon(zip(x_wrapped_minus_360, y, strict=True))

    else:
        # If dateline is not crossed, treat input poly as list
        polys = [poly]

    return polys


def polygon_from_bounding_box(bounding_box, margin_in_km):
    """Create a polygon (EPSG:4326) from the lat/lon `bounding_box`.

    Parameters
    ----------
    bounding_box : list
        Bounding box with lat/lon coordinates (decimal degrees) in the form of
        [West, South, East, North].
    margin_in_km : float
        Margin in kilometers to be added to the resultant polygon.

    Returns
    -------
    poly: shapely.Geometry.Polygon
        Bounding polygon corresponding to the provided bounding box with
        margin applied.

    """
    lon_min, lat_min, lon_max, lat_max = bounding_box
    logger.info(bounding_box)
    # note we can also use the center lat here
    lat_worst_case = max([lat_min, lat_max])

    # convert margin to degree
    lat_margin = margin_km_to_deg(margin_in_km)
    lon_margin = margin_km_to_longitude_deg(margin_in_km, lat=lat_worst_case)

    # Check if the bbox crosses the antimeridian and apply the margin accordingly
    # so that any resultant DEM is split properly by check_dateline
    if lon_max - lon_min > 180:
        lon_min, lon_max = lon_max, lon_min

    poly = box(
        lon_min - lon_margin,
        max([lat_min - lat_margin, -90]),
        lon_max + lon_margin,
        min([lat_max + lat_margin, 90]),
    )

    return poly


def download_map(polys, outfile: Path) -> list[Path]:
    """Download a map subregion corresponding to the provided polygon(s).

    Parameters
    ----------
    polys : list of shapely.geometry.Polygon
        List of polygons comprising the sub-regions of the global map to download.
    map_bucket : str
        Name of the S3 bucket containing the global map.
    map_vrt_key : str
        S3 key path to the location of the global map VRT within the
        bucket.
    outfile : str
        Path to where the output map VRT (and corresponding tifs) will be staged.

    Returns
    -------
    list[Path]
        Paths to GeoTiff files that the `outfile` VRT points to.

    """
    # Download the map for each provided Polygon
    region_list = []
    logger.info(f"Creating water mask to {outfile}")

    file_root = outfile.parent / outfile.stem
    output_tifs: list[Path] = []
    for idx, poly in enumerate(polys):
        vrt_filename = f"/vsis3/{MASK_S3_URL.replace('s3://', '')}"
        output_path = f"{file_root}_{idx}.tif"
        region_list.append(output_path)

        x_min, y_min, x_max, y_max = poly.bounds

        logger.info(
            f"Translating map for projection window "
            f"{[x_min, y_max, x_max, y_min]} to {output_path}"
        )

        ds = gdal.Open(vrt_filename, gdal.GA_ReadOnly)

        gdal.Translate(
            output_path, ds, format="GTiff", projWin=[x_min, y_max, x_max, y_min]
        )
        output_tifs.append(Path(output_path))

    # Build VRT with downloaded sub-regions
    gdal.BuildVRT(outfile, region_list)
    logger.info(f"Done, ancillary maps stored locally to {output_tifs}")
    return output_tifs


def create_water_mask(
    frame_id: int | None,
    bbox: tuple[float, float, float, float] | None = None,
    output: Path = Path("water_binary_mask.tif"),
    margin: int = 5,
    land_buffer: int = 1,
    ocean_buffer: int = 1,
) -> None:
    """Execute ancillary map staging."""
    # Make sure that output file has VRT extension
    if frame_id is None and bbox is None:
        raise ValueError("Must pass frame_id or bbox")
    if frame_id is not None:
        epsg, bounds = get_frame_bbox(frame_id=frame_id)

        t = Transformer.from_crs(epsg, 4326, always_xy=True)
        bbox = t.transform_bounds(*bounds)

    # Check connection to the S3 bucket
    logger.info(f"Checking connection to AWS S3 for {MASK_S3_URL}")
    p = S3Path(MASK_S3_URL)
    assert p.exists()

    # Derive the region polygon from the provided bounding box
    logger.info("Determining polygon from bounding box")
    poly = polygon_from_bounding_box(bbox, margin)

    # Check dateline crossing
    polys = check_dateline(poly)
    # Download the map for each polygon region and assemble them into a
    # single output VRT file
    out_dir = output.parent
    temp_vrt = out_dir / "water_mask.vrt"
    download_map(polys, temp_vrt)
    create_mask_from_distance(
        water_distance_file=temp_vrt,
        output_file=output,
        land_buffer=land_buffer,
        ocean_buffer=ocean_buffer,
    )


def create_mask_from_distance(
    water_distance_file: PathOrStr,
    output_file: PathOrStr,
    land_buffer: int = 0,
    ocean_buffer: int = 0,
) -> None:
    """Create a binary mask from the NISAR water distance mask with buffer zones.

    This function reads a water distance file, converts it to a binary mask
    with consideration for buffer zones, and writes the result to a new file.

    Parameters
    ----------
    water_distance_file : PathOrStr
        Path to the input water distance file.
    output_file : PathOrStr
        Path to save the output binary mask file.
    land_buffer : int, optional
        Buffer distance (in km) for land pixels. Only pixels this far or farther
        from water will be considered land. Default is 0.
    ocean_buffer : int, optional
        Buffer distance (in km) for ocean pixels. Only pixels this far or farther
        from land will be considered water. Default is 0.

    Notes
    -----
    Format of `water_distance_file` is UInt8, where:
    - 0 means "land"
    - 1 - 99 are ocean water pixels. The value is the distance (in km) to the shore.
      Value is rounded up to the nearest integer.
    - 100 - 200 are inland water pixels. Value is the distance to land.

    Output is a mask where 0 represents water pixels ("bad" pixels to ignore during
    processing/unwrapping), and 1 are land pixels to use.

    The buffer arguments make the masking more conservative. For example, a land_buffer
    of 2 means only pixels 2 km or farther from water will be masked as land. This helps
    account for potential changes in water levels.

    """
    # Load the water distance data
    water_distance_data = load_gdal(water_distance_file, masked=True)

    binary_mask = convert_distance_to_binary(
        water_distance_data, land_buffer, ocean_buffer
    )

    write_arr(
        arr=binary_mask.astype(np.uint8),
        output_name=output_file,
        like_filename=water_distance_file,
        dtype="uint8",
        nodata=255,
    )


def convert_distance_to_binary(
    water_distance_data: np.ma.MaskedArray, land_buffer: int = 0, ocean_buffer: int = 0
) -> np.ndarray:
    """Convert water distance data to a binary mask considering buffer zones.

    Parameters
    ----------
    water_distance_data : np.ma.MaskedArray
        Input water distance data as a masked array.
    land_buffer : int, optional
        Buffer distance (in km) for land pixels. Only pixels this far or farther
        from water will be considered land. Default is 0.
    ocean_buffer : int, optional
        Buffer distance (in km) for ocean pixels. Only pixels this far or farther
        from land will be considered water. Default is 0.

    Returns
    -------
    np.ndarray
        Binary mask where True represents land pixels and False represents water pixels.

    Notes
    -----
    The function applies the following logic:
    - Starts with all pixels as land (True).
    - Masks inland water pixels as water (False) if they are farther from land
        than the land_buffer.
    - Masks ocean pixels as water (False) if they are farther from shore than
        `ocean_buffer`.

    """
    # Create the binary mask with buffer considerations. Start all on (assume all land)
    binary_mask = np.ma.MaskedArray(
        np.ones_like(water_distance_data, dtype=bool), mask=water_distance_data.mask
    )

    # Mask inland water pixels (considering land buffer): anything 101 or higher is land
    inland_water_mask = water_distance_data > land_buffer + 100
    binary_mask[inland_water_mask] = False
    # For ocean, only look at values 1-100, then consider buffer
    ocean_water_mask = (water_distance_data <= 100) & (
        water_distance_data > ocean_buffer
    )
    binary_mask[ocean_water_mask] = False
    # Erode away small single-pixels
    closed_mask = ndimage.binary_closing(
        binary_mask.filled(0), structure=np.ones((3, 3)), border_value=1
    )
    return closed_mask


def create_layover_shadow_masks(
    cslc_static_files: Sequence[Filename],
    output_dir: Filename,
) -> list[Path]:
    """Create binary masks from the layover shadow CSLC static files.

    In the outputs, 0 indicates a bad masked pixel, 1 is a good pixel.

    Parameters
    ----------
    cslc_static_files : Sequence[Filename]
        List of CSLC static layer files to process
    output_dir : Filename
        Directory where output masks will be saved

    Returns
    -------
    list[Path]
        List of paths to the created binary layover shadow mask files

    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    output_files = []

    for burst_id, files in group_by_burst(cslc_static_files).items():
        if len(files) > 1:
            logger.warning(f"Found multiple static files for {burst_id}: {files}")
        f = files[0]
        input_name = format_nc_filename(f, ds_name="/data/layover_shadow_mask")
        out_file = output_path / f"layover_shadow_{burst_id}.tif"
        if out_file.exists():
            output_files.append(out_file)
            continue

        logger.info(f"Extracting layover shadow mask from {f} to {out_file}")
        layover_data = load_gdal(input_name)
        # we'll ignore the nodata region to be conservative
        layover_data[layover_data == 127] = 0
        not_layover_pixels = layover_data == 0
        write_arr(
            arr=not_layover_pixels,
            output_name=out_file,
            like_filename=input_name,
            nodata=127,
        )

        output_files.append(out_file)

    return output_files
