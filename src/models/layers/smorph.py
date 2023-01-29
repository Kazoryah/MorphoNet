"""Layer implementing the SMorph function."""

from typing import Any, Optional, Tuple, List
import torch
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import AxesDivider
from matplotlib.axes._axes import Axes
import numpy as np

from models.layers import PAD_MODE
from misc.utils import rmse
from .utils import init_context, folded_normal_
from .base import BaseLayer

# TODO shared param, alpha and/or filter


class SMorph(BaseLayer):
    """Module implementing the SMorph function."""

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

        # TODO try pin mem etc. for speed
        self.filter = nn.Parameter(
            torch.empty(
                (out_channels, in_channels, *self.filter_shape), **kwargs
            )
        )
        self.alpha = nn.Parameter(
            torch.empty((out_channels, in_channels), **kwargs)
        )

        self.init_parameters()

    def init_parameters(self) -> None:
        """Initialize tensors."""
        with torch.no_grad():
            init_context(folded_normal_, tensor=self.filter, mean=0.0, std=0.01)
            self.alpha.zero_()

    # TODO test fold, unfold, function results shapes
    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        input_padded = nn.functional.pad(batch, self.pad, mode=PAD_MODE)

        unfolder_ = nn.Unfold(kernel_size=self.filter_shape)
        unfolded = unfolder_(input_padded)

        sum_ = unfolded.transpose(1, 2) + self.filter.squeeze().ravel()
        sum_alpha = self.alpha.squeeze() * sum_
        exp_sum_alpha = sum_alpha.exp()
        sum_exp_sum_alpha = sum_ * exp_sum_alpha

        result = sum_exp_sum_alpha.sum(2) / exp_sum_alpha.sum(2)

        return result.view(*batch.size())

    def plot_(
        self,
        axis: Axes,
        cmap: str = "plasma",
        target: Optional[np.ndarray] = None,
        comments: str = "",
        divider: Optional[AxesDivider] = None,
        limits: Optional[List[Tuple[float, float]]] = None,
    ) -> Axes:  # pragma: no cover
        alpha = self.alpha.squeeze().detach().cpu()
        if alpha < 0:
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
        if limits is not None and len(limits) >= 2:
            if alpha < 0:
                plot.set_clim(limits[0])
            else:
                plot.set_clim(limits[1])
        if divider is None:
            divider = make_axes_locatable(axis)
        clb_ax = divider.append_axes("right", size="5%", pad=0.05)
        clb_ax.set_box_aspect(15)
        plt.colorbar(plot, cax=clb_ax)

        axis.set_title(r"$\alpha$: " + f"{alpha:.3f}", fontsize=20)
        if target is not None:
            rmse_ = rmse(filter_.numpy() * invert, target)
            comments = f"RMSE: {rmse_:.3f}\n{comments}"

        axis.set_xlabel(comments, fontsize=20)

        return axis


class SMorphTanh(SMorph):
    """Module implementing the SMorph function with tanh modification."""

    # TODO test fold, unfold, function results shapes
    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        input_padded = nn.functional.pad(batch, self.pad, mode=PAD_MODE)

        unfolder_ = nn.Unfold(kernel_size=self.filter_shape)
        unfolded = unfolder_(input_padded)

        sum_ = unfolded.transpose(1, 2) + (
            torch.tanh(self.alpha) * self.filter.squeeze().ravel()
        )
        sum_alpha = self.alpha.squeeze() * sum_
        exp_sum_alpha = sum_alpha.exp()
        sum_exp_sum_alpha = sum_ * exp_sum_alpha

        result = sum_exp_sum_alpha.sum(2) / exp_sum_alpha.sum(2)

        return result.view(*batch.size())

    def plot_(
        self,
        axis: Axes,
        cmap: str = "plasma",
        target: Optional[np.ndarray] = None,
        comments: str = "",
        divider: Optional[AxesDivider] = None,
        limits: Optional[List[Tuple[float, float]]] = None,
    ) -> Axes:  # pragma: no cover
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

        axis.set_title(
            r"$\alpha$: " + f"{self.alpha.squeeze().detach().cpu():.3f}",
            fontsize=20,
        )
        if target is not None:
            comments = f"RMSE: {rmse(filter_.numpy(), target):.3f}\n{comments}"

        axis.set_xlabel(comments, fontsize=20)

        return axis


# TODO not tested
class SMorphDual(BaseLayer):  # pylint: disable=abstract-method
    """Module implementing two SMorph layers with shared filter weights."""

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
        self.alpha1 = nn.Parameter(
            torch.empty((out_channels, in_channels), **kwargs)
        )
        self.alpha2 = nn.Parameter(
            torch.empty((out_channels, in_channels), **kwargs)
        )

        self.init_parameters()

    def init_parameters(self) -> None:
        """Initialize tensors."""
        with torch.no_grad():
            init_context(folded_normal_, tensor=self.filter, mean=0.0, std=0.01)
            self.alpha1.zero_()
            self.alpha2.zero_()

    # TODO test fold, unfold, function results shapes
    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        input_padded = nn.functional.pad(batch, self.pad, mode=PAD_MODE)

        unfolder_ = nn.Unfold(kernel_size=self.filter_shape)
        unfolded = unfolder_(input_padded)

        sum_ = unfolded.transpose(1, 2) + self.filter.squeeze().ravel()
        sum_alpha = self.alpha1.squeeze() * sum_
        exp_sum_alpha = sum_alpha.exp()
        sum_exp_sum_alpha = sum_ * exp_sum_alpha

        result = sum_exp_sum_alpha.sum(2) / exp_sum_alpha.sum(2)

        input_padded = nn.functional.pad(
            result.view(*batch.size()), self.pad, mode=PAD_MODE
        )

        unfolder_ = nn.Unfold(kernel_size=self.filter_shape)
        unfolded = unfolder_(input_padded)

        sum_ = unfolded.transpose(1, 2) + self.filter.squeeze().ravel()
        sum_alpha = self.alpha2.squeeze() * sum_
        exp_sum_alpha = sum_alpha.exp()
        sum_exp_sum_alpha = sum_ * exp_sum_alpha

        result = sum_exp_sum_alpha.sum(2) / exp_sum_alpha.sum(2)

        return result.view(*batch.size())
