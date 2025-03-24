from typing import Optional, Sequence

from darts.models import ARIMA


class ARIMAFix(ARIMA):
    @property
    def extreme_lags(
        self,
    ) -> tuple[
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
        int,
        Optional[int],
    ]:
        if not self._fit_called:
            # IIRC, the current extreme_lags implementation is just for training, not for inference
            return super().extreme_lags

        p, d, q = self.order
        if self.seasonal_order and not self.seasonal_order == (0, 0, 0, 0):
            raise NotImplementedError("Seasonal ARIMA not supported for fixed extreme_lags yet")

        if isinstance(p, Sequence):
            p = max(p)

        if isinstance(q, Sequence):
            q = max(q)

        # when diffing, you look between t and t-1, so d=1 means you look back one for all potential lags,
        # and d=2, you look at diff like this (t - (t-1)) - ((t-1) - (t-2)), so looking back 2.
        # ergo, I think it must be added to the total.
        max_lookback = max(p, q) + d

        return -max_lookback, -1, None, None, 0, 0, 0, None
