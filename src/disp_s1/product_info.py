from dataclasses import dataclass

import numpy as np
from dolphin import io
from dolphin._types import Filename
from numpy.typing import ArrayLike, DTypeLike


@dataclass
class ProductInfo:
    name: str
    description: str
    fillvalue: DTypeLike
    attrs: dict


unwrapped_phase_info = ProductInfo(
    name="unwrapped_phase",
    description="Unwrapped phase",
    fillvalue=np.nan,
    attrs=dict(units="radians"),
)

connected_component_labels = ProductInfo(
    name="connected_component_labels",
    description="Connected component labels of the unwrapped phase",
    fillvalue=0,
    attrs=dict(units="unitless"),
)

temporal_coherence = ProductInfo(
    name="temporal_coherence",
    description="Temporal coherence of phase inversion",
    fillvalue=np.nan,
    attrs=dict(units="unitless"),
)

interferometric_correlation = ProductInfo(
    name="interferometric_correlation",
    description=(
        "Estimate of interferometric correlation derived from"
        " multilooked interferogram."
    ),
    fillvalue=np.nan,
    attrs=dict(units="unitless"),
)

persistent_scattere_mask = ProductInfo(
    name="persistent_scatterer_mask",
    description=(
        "Mask of persistent scatterers downsampled to the multilooked output grid."
    ),
    fillvalue=255,
    attrs=dict(units="unitless"),
)


class DispProductLayer:
    def __init__(
        self, filename: Filename, product_info: ProductInfo, masked: bool = False
    ):
        self.filename = filename
        self.product_info = product_info
        self.data: ArrayLike = io.load_gdal(filename, masked=masked)


class UnwrappedPhaseLayer(DispProductLayer):
    def __init__(self, filename: Filename):
        super().__init__(
            filename=filename, product_info=unwrapped_phase_info, masked=True
        )
        self.data = np.ma.filled(self.data, 0)


class ConnectedComponentLabelsLayer(DispProductLayer):
    def __init__(self, filename: Filename):
        super().__init__(
            filename=filename, product_info=unwrapped_phase_info, masked=True
        )
