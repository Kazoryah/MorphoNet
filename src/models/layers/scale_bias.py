"""Layer implementing affine function."""

from typing import Any, Optional, Tuple, List
import torch
from torch import nn

from mpl_toolkits.axes_grid1.axes_divider import AxesDivider
from matplotlib.axes._axes import Axes
import numpy as np

from .base import BaseLayer


class ScaleBias(BaseLayer):
    """Simple affine layer."""

    def __init__(self, num_features: int, **kwargs: Any) -> None:
        """Initialize affine ScaleBias Layer."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features, 1, 1, **kwargs))
        self.bias = nn.Parameter(torch.zeros(num_features, 1, 1, **kwargs))

    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        return batch * self.weight + self.bias

    def plot_(
        self,
        axis: Axes,
        cmap: str = "plasma",
        target: Optional[np.ndarray] = None,
        comments: str = "",
        divider: Optional[AxesDivider] = None,
        limits: Optional[List[Tuple[float, float]]] = None,
    ) -> Axes:  # pragma: no cover

        axis.get_yaxis().set_ticks([])
        axis.get_xaxis().set_ticks([])
        axis.set_box_aspect(1)

        text = f"Scale: {self.weight.squeeze().detach().cpu():.3f}\n"
        text += f"Bias: {self.bias.squeeze().detach().cpu():.3f}"

        axis.text(0.5, 0.5, text, ha="center", fontsize=30)

        return axis
