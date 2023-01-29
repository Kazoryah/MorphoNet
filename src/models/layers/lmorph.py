"""Layer implementing the LMorph function."""

from typing import Any, Optional, Tuple, List
from torch import nn
import torch
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import AxesDivider
import numpy as np

from misc.utils import rmse
from .utils import init_context, folded_normal_
from .base import BaseLayer

PAD_MODE = "reflect"

# TODO check dual lmorph EqMax utility
# TODO check if filter is well clamped at the right time


class LMorph(BaseLayer):
    """Module implementing the LMorph function."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_size: int,
        **kwargs: Any,
    ):
        super().__init__()
        self.filter_shape = (filter_size, filter_size)
        self.pad_h = self.filter_shape[0] // 2
        self.pad_w = self.filter_shape[1] // 2
        self.pad = (self.pad_w, self.pad_w, self.pad_h, self.pad_h)

        self.filter = nn.Parameter(
            torch.empty(
                (out_channels, in_channels, *self.filter_shape), **kwargs
            )
        )
        self.p = nn.Parameter(  # pylint: disable=invalid-name
            torch.empty((out_channels, in_channels), **kwargs)
        )

        self.init_parameters()

    def init_parameters(self) -> None:
        """Initialize tensors."""
        with torch.no_grad():
            init_context(folded_normal_, tensor=self.filter, mean=0.0, std=0.01)
            self.p.zero_()

    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        with torch.no_grad():
            self.filter.clamp_min_(0.0)

        input_padded = nn.functional.pad(batch, self.pad, mode=PAD_MODE)

        imin = input_padded.min().detach()
        imax = input_padded.max().detach()
        input_padded = 1.0 + (input_padded - imin) / (imax - imin)

        unfolder_ = nn.Unfold(kernel_size=self.filter_shape)
        unfolded = unfolder_(input_padded)

        sum_ = unfolded.transpose(1, 2) + self.filter.squeeze().ravel()
        pow1 = sum_.pow(self.p + 1)
        pow2 = sum_.pow(self.p)

        result = pow1.sum(2) / pow2.sum(2)

        return result.view(*batch.size()) - 1.0

    def plot_(
        self,
        axis: Axes,
        cmap: str = "plasma",
        target: Optional[np.ndarray] = None,
        comments: str = "",
        divider: Optional[AxesDivider] = None,
        limits: Optional[List[Tuple[float, float]]] = None,
    ) -> Axes:  # pragma: no cover
        p = self.p.squeeze().detach().cpu()  # pylint: disable=invalid-name
        if p < 0:
            cmap = "plasma_r"
            invert = -1
        else:
            invert = 1

        axis.invert_yaxis()
        axis.get_yaxis().set_ticks([])
        axis.get_xaxis().set_ticks([])
        axis.set_box_aspect(1)

        filter_ = self.filter.squeeze().detach().cpu()

        plot = axis.pcolormesh(filter_, cmap=cmap)
        if divider is None:
            divider = make_axes_locatable(axis)
        clb_ax = divider.append_axes("right", size="5%", pad=0.05)
        clb_ax.set_box_aspect(15)
        plt.colorbar(plot, cax=clb_ax)

        axis.set_title(r"$p$: " + f"{p:.3f}", fontsize=20)
        if target is not None:
            rmse_ = rmse(filter_.numpy() * invert, target)
            comments = f"RMSE: {rmse_:.3f}\n{comments}"

        axis.set_xlabel(comments, fontsize=20)

        return axis
