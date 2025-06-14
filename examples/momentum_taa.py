import operator
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import pytz

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from data.data_loader import download_sector_etf_data
from qstrader.alpha_model.alpha_model import AlphaModel
from qstrader.alpha_model.fixed_signals import FixedSignalsAlphaModel
from qstrader.asset.equity import Equity
from qstrader.asset.universe.dynamic import DynamicUniverse
from qstrader.asset.universe.static import StaticUniverse
from qstrader.signals.momentum import MomentumSignal
from qstrader.signals.signals_collection import SignalsCollection
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.data.daily_bar_csv import CSVDailyBarDataSource
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading.backtest import BacktestTradingSession


class TopNMomentumAlphaModel(AlphaModel):

    def __init__(
        self, signals, mom_lookback, mom_top_n, universe, data_handler
    ):
        """
        Initialise the TopNMomentumAlphaModel

        Parameters
        ----------
        signals : `SignalsCollection`
            The entity for interfacing with various pre-calculated
            signals. In this instance we want to use 'momentum'.
        mom_lookback : `integer`
            The number of business days to calculate momentum
            lookback over.
        mom_top_n : `integer`
            The number of assets to include in the portfolio,
            ranking from highest momentum descending.
        universe : `Universe`
            The collection of assets utilised for signal generation.
        data_handler : `DataHandler`
            The interface to the CSV data.

        Returns
        -------
        None
        """
        self.signals = signals
        self.mom_lookback = mom_lookback
        self.mom_top_n = mom_top_n
        self.universe = universe
        self.data_handler = data_handler

    def _highest_momentum_asset(self, dt):
        """
        Calculate momentum for the current universe assets
        """
        valid_assets = self.universe.get_assets(dt)
        if not valid_assets:
            return []
        
        # Calculate momentum for valid assets
        all_momenta = {}
        for asset in valid_assets:
            try:
                momentum = self.signals['momentum'](asset, self.mom_lookback)
                if not np.isnan(momentum):
                    all_momenta[asset] = momentum
            except:  # Skip any assets that cause errors
                continue

        # Return empty list if no valid momentum scores
        if not all_momenta:
            return []

        # Sort assets by momentum score
        sorted_assets = sorted(
            all_momenta.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        # Only go to cash if all momentum scores are deeply negative
        top_momentum = sorted_assets[0][1]
        if len(sorted_assets) == 0 or top_momentum < -0.05:  # 5% threshold
            if 'EQ:SHY' in valid_assets:
                return ['EQ:SHY']
            return []

        # Return top N assets with positive momentum
        return [
            asset for asset, momentum in sorted_assets
            if momentum > 0
        ][:self.mom_top_n]

    def _generate_signals(
        self, dt, weights
    ):
        """
        Calculate the highest performing momentum for each
        asset then assign 1 / N of the signal weight to each
        of these assets.

        Parameters
        ----------
        dt : `pd.Timestamp`
            The datetime for which the signal weights
            should be calculated.
        weights : `dict{str: float}`
            The current signal weights dictionary.

        Returns
        -------
        `dict{str: float}`
            The newly created signal weights dictionary.
        """
        top_assets = self._highest_momentum_asset(dt)
        num_assets = len(top_assets)
        
        if num_assets > 0:
            weight = 1.0 / num_assets
            for asset in top_assets:
                weights[asset] = weight
        
        return weights

    def __call__(self, dt):
        assets = self.universe.get_assets(dt)
        weights = {asset: 0.0 for asset in assets}

        # Only generate weights if the current time exceeds the
        # momentum lookback period
        if self.signals.warmup >= self.mom_lookback:
            weights = self._generate_signals(dt, weights)
        return weights


if __name__ == "__main__":
    # Set up backtest parameters first
    print("Setting up backtest parameters...")
    
    # Model parameters (adjusted for better performance)
    mom_lookback = 63  # Three months of business days
    mom_top_n = 3  # Select top 3 performing assets
    use_vol_adj = False  # Disable volatility adjustment
    use_ema = True  # Use exponential moving average
    ema_alpha = 0.97  # Slower decay for more stable signals

    # Use more recent dates including XLC inception
    start_dt = pd.Timestamp('2018-01-01 14:30:00', tz=pytz.UTC)
    burn_in_dt = pd.Timestamp('2018-02-01 14:30:00', tz=pytz.UTC)
    end_dt = pd.Timestamp('2024-05-31 23:59:00', tz=pytz.UTC)
    
    # Download or update ETF data
    print("Downloading/updating ETF data...")
    if not download_sector_etf_data():
        print("Error downloading some ETF data. The strategy will continue with available data.")

    print("Creating dynamic asset universe...")

    # Define all sector ETFs including XLC
    base_sectors = "BCEFIKPUVY"  # Original SPDR sectors
    asset_dates = {
        **{f'EQ:XL{s}': start_dt for s in base_sectors},  # Original sectors
        'EQ:XLRE': start_dt,  # Real Estate
        'EQ:SHY': start_dt,   # Treasury Bond (Safe Haven)
        'EQ:XLC': pd.Timestamp('2018-06-18 00:00:00', tz=pytz.UTC)  # Communication Services
    }
    
    # Create list of symbols from asset keys
    strategy_symbols = sorted([asset.split(':')[1] for asset in asset_dates.keys()])
    strategy_universe = DynamicUniverse(asset_dates)

    # To avoid loading all CSV files in the directory, set the
    # data source to load only those provided symbols
    csv_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    strategy_data_source = CSVDailyBarDataSource(csv_dir, Equity, adjust_prices=False, csv_symbols=strategy_symbols)
    strategy_data_handler = BacktestDataHandler(strategy_universe, data_sources=[strategy_data_source])

    # Initialize momentum signal for end-of-month rebalancing
    momentum = MomentumSignal(
        start_dt,
        strategy_universe,
        lookbacks=[mom_lookback],
        use_vol_adj=use_vol_adj,
        use_ema=use_ema,
        ema_alpha=ema_alpha
    )
    signals = SignalsCollection({'momentum': momentum}, strategy_data_handler)

    # Generate the alpha model instance
    strategy_alpha_model = TopNMomentumAlphaModel(
        signals, mom_lookback, mom_top_n,
        strategy_universe, strategy_data_handler
    )

    # Create and run strategy backtest
    strategy_backtest = BacktestTradingSession(
        start_dt,
        end_dt,
        strategy_universe,
        strategy_alpha_model,
        signals=signals,
        rebalance='end_of_month',
        long_only=True,
        cash_buffer_percentage=0.01,
        initial_cash=1000000.0,  # Set initial cash explicitly
        burn_in_dt=burn_in_dt,
        data_handler=strategy_data_handler
    )
    strategy_backtest.run()

    # Construct benchmark assets (buy & hold SPY)
    benchmark_symbols = ['SPY']
    benchmark_assets = ['EQ:SPY']
    benchmark_universe = StaticUniverse(benchmark_assets)
    benchmark_data_source = CSVDailyBarDataSource(csv_dir, Equity, adjust_prices=False, csv_symbols=benchmark_symbols)
    benchmark_data_handler = BacktestDataHandler(benchmark_universe, data_sources=[benchmark_data_source])

    # Construct a benchmark Alpha Model that provides
    # 100% static allocation to the SPY ETF, with no rebalance
    benchmark_alpha_model = FixedSignalsAlphaModel({'EQ:SPY': 1.0})
    benchmark_backtest = BacktestTradingSession(
        burn_in_dt,  # Use burn_in_dt for benchmark start
        end_dt,
        benchmark_universe,
        benchmark_alpha_model,
        rebalance='buy_and_hold',
        long_only=True,
        cash_buffer_percentage=0.01,
        data_handler=benchmark_data_handler
    )
    benchmark_backtest.run()

    # Performance Output
    # Create compact tearsheet subclass
    class CompactTearsheet(TearsheetStatistics):
        def plot_results(self, filename=None):
            # Override only the figure creation part
            vertical_sections = 5
            # Use small figure size
            fig = plt.figure(figsize=(16, 7.5))  # 1600x800 pixels at 100 DPI
            fig.suptitle(self.title, y=0.94, weight='bold')
            gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.25, hspace=0.5)

            # Use parent class methods for all plotting
            stats = self.get_results(self.strategy_equity)
            bench_stats = None
            if self.benchmark_equity is not None:
                bench_stats = self.get_results(self.benchmark_equity)

            ax_equity = plt.subplot(gs[:2, :])
            ax_drawdown = plt.subplot(gs[2, :])
            ax_monthly_returns = plt.subplot(gs[3, :2])
            ax_yearly_returns = plt.subplot(gs[3, 2])
            ax_txt_curve = plt.subplot(gs[4, 0])

            # Call parent plotting methods
            self._plot_equity(stats, bench_stats=bench_stats, ax=ax_equity)
            self._plot_drawdown(stats, ax=ax_drawdown)
            self._plot_monthly_returns(stats, ax=ax_monthly_returns)
            self._plot_yearly_returns(stats, ax=ax_yearly_returns)
            self._plot_txt_curve(stats, bench_stats=bench_stats, ax=ax_txt_curve)

            if filename:
                fig.savefig(filename)
            plt.show()

    # Create and display tearsheet using compact subclass
    tearsheet = CompactTearsheet(
        strategy_equity=strategy_backtest.get_equity_curve(),
        benchmark_equity=benchmark_backtest.get_equity_curve(),
        title='US Sector Momentum - Single Best Sector (Vol-Adj)'
    )
    
    tearsheet.plot_results()
