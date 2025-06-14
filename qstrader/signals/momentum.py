import numpy as np
import pandas as pd

from qstrader.signals.signal import Signal
from qstrader.signals.vol import VolatilitySignal


class MomentumSignal(Signal):
    """
    Indicator class to calculate lookback-period momentum
    (based on cumulative return of last N periods) for
    a set of prices.

    If the number of available returns is less than the
    lookback parameter the momentum is calculated on
    this subset.

    Parameters
    ----------
    start_dt : `pd.Timestamp`
        The starting datetime (UTC) of the signal.
    universe : `Universe`
        The universe of assets to calculate the signals for.
    lookbacks : `list[int]`
        The number of lookback periods to store prices for.
    """

    def __init__(self, start_dt, universe, lookbacks, use_vol_adj=True, use_ema=False, ema_alpha=0.94):
        """
        Parameters
        ----------
        start_dt : `pd.Timestamp`
            The starting datetime (UTC) of the signal.
        universe : `Universe`
            The universe of assets to calculate the signals for.
        lookbacks : `list[int]`
            The number of lookback periods to store prices for.
        use_vol_adj : `bool`
            Whether to adjust momentum by volatility.
        use_ema : `bool`
            Whether to use exponential moving average for returns.
        ema_alpha : `float`
            The decay factor for EMA calculation.
        """
        bumped_lookbacks = [lookback + 1 for lookback in lookbacks]
        super().__init__(start_dt, universe, bumped_lookbacks)
        self.vol_signal = VolatilitySignal(start_dt, universe, lookbacks)
        self.use_vol_adj = use_vol_adj
        self.use_ema = use_ema
        self.ema_alpha = ema_alpha

    @staticmethod
    def _asset_lookback_key(asset, lookback):
        """
        Create the buffer dictionary lookup key based
        on asset name and lookback period.

        Parameters
        ----------
        asset : `str`
            The asset symbol name.
        lookback : `int`
            The lookback period.

        Returns
        -------
        `str`
            The lookup key.
        """
        return '%s_%s' % (asset, lookback + 1)

    def _cumulative_return(self, asset, lookback):
        """
        Calculate the cumulative returns for the provided
        lookback period ('momentum') based on the price
        buffers for a particular asset.

        Parameters
        ----------
        asset : `str`
            The asset symbol name.
        lookback : `int`
            The lookback period.

        Returns
        -------
        `float`
            The cumulative return ('momentum') for the period.
        """
        series = pd.Series(
            self.buffers.prices[MomentumSignal._asset_lookback_key(asset, lookback)]
        )
        returns = series.pct_change().dropna()

        if len(returns) < 1:
            return 0.0

        if self.use_ema:
            # Apply exponential moving average to returns
            returns = returns.ewm(alpha=self.ema_alpha).mean()

        returns = returns.to_numpy()
        momentum = (np.cumprod(1.0 + np.array(returns)) - 1.0)[-1]

        if self.use_vol_adj:
            volatility = self.vol_signal(asset, lookback)
            if volatility > 0:
                return momentum / volatility
            return 0.0
        
        return momentum

    def __call__(self, asset, lookback):
        """
        Calculate the lookback-period momentum
        for the asset.

        Parameters
        ----------
        asset : `str`
            The asset symbol name.
        lookback : `int`
            The lookback period.

        Returns
        -------
        `float`
            The momentum for the period.
        """
        return self._cumulative_return(asset, lookback)
