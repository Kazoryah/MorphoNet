"""One layer network with SMorph function."""

from typing import Callable, Any
import torch

from .base import BaseNetwork
from .layers.scale_bias import ScaleBias
from .layers.smorph import SMorph, SMorphTanh
from .utils import NotTested


class SMorphNet(BaseNetwork):
    """Network with one SMorph layer."""

    def __init__(
        self,
        filter_size: int,
        loss_function: Callable,
        **kwargs: Any,
    ):
        super().__init__(loss_function=loss_function)
        self._set_hparams(
            {
                "filter_size": filter_size,
                "loss_function": loss_function,
                **kwargs,
            }
        )

        self.sm1 = SMorph(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.sb1 = ScaleBias(num_features=1, **kwargs)

    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        batch = self.sm1(batch)
        batch = self.sb1(batch)

        return batch


class SMorphNetTanh(BaseNetwork):
    """Network with one SMorphTanh layer."""

    def __init__(
        self,
        filter_size: int,
        loss_function: Callable,
        **kwargs: Any,
    ):
        super().__init__(loss_function=loss_function)
        self._set_hparams(
            {
                "filter_size": filter_size,
                "loss_function": loss_function,
                **kwargs,
            }
        )

        self.sm1 = SMorphTanh(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.sb1 = ScaleBias(num_features=1, **kwargs)

    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        batch = self.sm1(batch)
        batch = self.sb1(batch)

        return batch


class SMorphNetTanhDouble(BaseNetwork):
    """Network with two SMorphTanh layers."""

    def __init__(
        self,
        filter_size: int,
        loss_function: Callable,
        **kwargs: Any,
    ):
        super().__init__(loss_function=loss_function)
        self._set_hparams(
            {
                "filter_size": filter_size,
                "loss_function": loss_function,
                **kwargs,
            }
        )

        self.sm1 = SMorphTanh(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.sm2 = SMorphTanh(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.sb1 = ScaleBias(num_features=1, **kwargs)

    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        batch = self.sm1(batch)
        batch = self.sm2(batch)
        batch = self.sb1(batch)

        return batch





class SMorphNetDouble(BaseNetwork):
    """Network with two SMorph layers."""

    def __init__(
        self,
        filter_size: int,
        loss_function: Callable,
        **kwargs: Any,
    ):
        super().__init__(loss_function=loss_function)
        self._set_hparams(
            {
                "filter_size": filter_size,
                "loss_function": loss_function,
                **kwargs,
            }
        )

        self.sm1 = SMorph(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.sm2 = SMorph(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.sb1 = ScaleBias(num_features=1, **kwargs)

    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        batch = self.sm1(batch)
        batch = self.sm2(batch)
        batch = self.sb1(batch)

        return batch


class SMorphNetFour(BaseNetwork):
    """Network with four SMorph layers."""

    def __init__(
        self,
        filter_size: int,
        loss_function: Callable,
        **kwargs: Any,
    ):
        super().__init__(loss_function=loss_function)
        self._set_hparams(
            {
                "filter_size": filter_size,
                "loss_function": loss_function,
                **kwargs,
            }
        )

        self.sm1 = SMorph(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.sm2 = SMorph(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.sm3 = SMorph(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.sm4 = SMorph(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.sb1 = ScaleBias(num_features=1, **kwargs)

    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        batch = self.sm1(batch)
        batch = self.sm2(batch)
        batch = self.sm3(batch)
        batch = self.sm4(batch)
        batch = self.sb1(batch)

        return batch


# TODO not tested
class SMorphNetWTH(BaseNetwork, metaclass=NotTested):
    """
    Network with two SMorph layers and adaptation to support white top-hat
    operation.
    """

    def __init__(
        self,
        filter_size: int,
        loss_function: Callable,
        **kwargs: Any,
    ):
        super().__init__(loss_function=loss_function)
        self._set_hparams(
            {
                "filter_size": filter_size,
                "loss_function": loss_function,
                **kwargs,
            }
        )

        self.sm1 = SMorph(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.sm2 = SMorph(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.sb1 = ScaleBias(num_features=1, **kwargs)
        self.lmbda = torch.nn.Parameter(torch.empty((1), **kwargs))

    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        batch = self.sm1(batch)
        batch = self.sm2(batch)
        batch = (0.5 + torch.tanh(self.lmbda) / 2) * batch - torch.tanh(
            self.lmbda
        ) * batch
        batch = self.sb1(batch)

        return batch


class SMorphNetNoSB(BaseNetwork):
    """Network with only one SMorph layer (no scale bias layer)."""

    def __init__(
        self,
        filter_size: int,
        loss_function: Callable,
        **kwargs: Any,
    ):
        super().__init__(loss_function=loss_function)
        self._set_hparams(
            {
                "filter_size": filter_size,
                "loss_function": loss_function,
                **kwargs,
            }
        )

        self.sm1 = SMorph(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )

    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        batch = self.sm1(batch)

        return batch


class SMorphNetDoubleNoSB(BaseNetwork):
    """Network with only two SMorph layers (no scale bias layer)."""

    def __init__(
        self,
        filter_size: int,
        loss_function: Callable,
        **kwargs: Any,
    ):
        super().__init__(loss_function=loss_function)
        self._set_hparams(
            {
                "filter_size": filter_size,
                "loss_function": loss_function,
                **kwargs,
            }
        )

        self.sm1 = SMorph(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.sm2 = SMorph(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )

    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        batch = self.sm1(batch)
        batch = self.sm2(batch)

        return batch


class SMorphNetTanhDoubleNoSB(BaseNetwork):
    """Network with only two SMorphTanh layers."""

    def __init__(
        self,
        filter_size: int,
        loss_function: Callable,
        **kwargs: Any,
    ):
        super().__init__(loss_function=loss_function)
        self._set_hparams(
            {
                "filter_size": filter_size,
                "loss_function": loss_function,
                **kwargs,
            }
        )

        self.sm1 = SMorphTanh(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )
        self.sm2 = SMorphTanh(
            in_channels=1, out_channels=1, filter_size=filter_size, **kwargs
        )

    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ
        batch = self.sm1(batch)
        batch = self.sm2(batch)

        return batch
