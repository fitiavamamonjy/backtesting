import matplotlib.pyplot as plt
import pandas as pd
import ta
import pytz
from datetime import datetime
from dataclasses import dataclass, field
from models.candle import Candle

TZ = "America/New_York"
TIMEZONE = pytz.timezone(TZ)


@dataclass
class Backtest:
    # Mandatory fields
    data: pd.DataFrame

    # Optional fields
    commision: float = 3.24
    risk_free: float = 0.02 # taux sans risque (ex. : 2% annuel = 0.02)
    start_session_hour: float = 0
    start_session_minute: float = 0
    end_session_hour: float = 23
    end_session_minute: float = 59

    # These fields should not edit outside of this file
    initial_balance: float = 1000.0
    balance: float = 1000.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    trades: list = field(default_factory=list)
    position: dict = field(default_factory=dict)
    win_trades: list = field(default_factory=list)
    loss_trades: list = field(default_factory=list)
    current_candle: Candle = None
    current_index: int = 0
    profit_factor: float = 0.0
    long_profit_factor: float = 0.0
    short_profit_factor: float = 0.0
    ema: float = 100

    # These fields used for analysis. You can call theme manualy from jupyter notebook
    df_trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    df_days: pd.DataFrame = field(default_factory=pd.DataFrame)
    overview: pd.DataFrame = field(default_factory=pd.DataFrame)
    performance: pd.DataFrame = field(default_factory=pd.DataFrame)
    deep_analysis: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_analysis: pd.DataFrame = field(default_factory=pd.DataFrame)
    hourly_analysis: pd.DataFrame = field(default_factory=pd.DataFrame)
    risk_ratio: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self):
        self.data = self.get_data(self.data)
        self.balance = self.initial_balance

    #  ===========================================================================
    #                                   PROPERTY
    #  ===========================================================================
    @property
    def win_rates(self):
        """
        Calculate the win rate.
        """
        win = len(self.win_trades)
        total = len(self.trades)
        loss = len(self.loss_trades)
        win_rates = win / total if total > 0 else 0.0
        return f"{win_rates * 100:.2f}% {win}/{loss}"

    #  ===========================================================================
    #                                   INDICATORS
    #  ===========================================================================
    def _calculate_ema(self, period=200):
        """
        Calculate the Exponential Moving Average (EMA) for a given period.
        """  # EMA sur la clôture
        self.data["EMA_Close"] = ta.trend.ema_indicator(
            close=self.data["close"], window=period
        )

        # EMA sur le high
        self.data["EMA_High"] = ta.trend.ema_indicator(
            close=self.data["high"], window=period
        )

        # EMA sur le low
        self.data["EMA_Low"] = ta.trend.ema_indicator(
            close=self.data["low"], window=period
        )

    def _calculate_bollinger_bands(self, period=20, std_dev=2):
        """
        Calculate the Bollinger Bands for a given period and standard deviation.
        """
        bb = ta.volatility.BollingerBands(
            close=self.data["close"], window=period, window_dev=std_dev
        )
        self.data["bb_high"] = bb.bollinger_hband()
        self.data["bb_low"] = bb.bollinger_lband()
        self.data["bb_middle"] = bb.bollinger_mavg()

    def _calculate_stochastic(self, period=10, smooth_period=6):
        stochastic = ta.momentum.StochasticOscillator(
            high=self.data["high"],
            low=self.data["low"],
            close=self.data["close"],
            window=period,
            smooth_window=smooth_period,
        )
        stoch = stochastic.stoch()

        self.data["stoch"] = stoch
        self.data["stoch_overbought"] = stoch > 65
        self.data["stoch_oversold"] = stoch < 35

    def calculate_indicators(self):
        self._calculate_ema(self.ema)
        self._calculate_bollinger_bands()
        self._calculate_stochastic()

    #  ===========================================================================
    #                                   DATA
    #  ===========================================================================
    def get_data(self, data):
        no_variation = (data["high"] == data["low"]) & (data["low"] == data["close"])
        df = data[~no_variation].copy().reset_index(drop=True)
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["time"] = df["time"].dt.tz_localize("UTC").dt.tz_convert(TZ)
        return df

    def get_candle(self, index):
        """
        Get the candle data for a given index.
        """
        candle_data = self.data.iloc[index]
        return Candle(
            index=index,
            open=candle_data["open"],
            high=candle_data["high"],
            low=candle_data["low"],
            close=candle_data["close"],
            time=candle_data["time"],
            ema_high=candle_data["EMA_High"],
            ema_low=candle_data["EMA_Low"],
            bb_high=candle_data["bb_high"],
            bb_low=candle_data["bb_low"],
            stoch=candle_data["stoch"],
        )

    def get_session(self, start_hour, end_hour, start_min=0, end_min=0):
        """
        Get the session for the trades.
        """
        session_start = TIMEZONE.localize(
            datetime(
                self.current_candle.time.year,
                self.current_candle.time.month,
                self.current_candle.time.day,
                start_hour,
                start_min,
                0,
            )
        )
        session_end = TIMEZONE.localize(
            datetime(
                self.current_candle.time.year,
                self.current_candle.time.month,
                self.current_candle.time.day,
                end_hour,
                end_min,
                0,
            )
        )
        return session_start, session_end

    #  ===========================================================================
    #                                   TRADES
    #  ===========================================================================
    def take_trade(self, direction="buy"):
        """
        Take a trade.
        """
        if direction == "buy":
            # SL basé sur ema_low
            sl_price = (
                self.current_candle.low > self.current_candle.ema_low
                and self.current_candle.ema_low
                or self.current_candle.low - 0.05
            )
            risk = self.current_candle.close - sl_price
            tp_price = self.current_candle.close + (risk * 2)  # TP = RR 1:2
        elif direction == "sell":
            #  SL basé sur ema_high
            sl_price = (
                self.current_candle.high < self.current_candle.ema_high
                and self.current_candle.ema_high
                or self.current_candle.high + 0.05
            )
            risk = sl_price - self.current_candle.close
            tp_price = self.current_candle.close - (risk * 2)  # TP = RR 1:2
        else:
            raise ValueError("Invalid direction. Must be 'buy' or 'sell'.")

        self.position.update(
            day=self.current_candle.time.date(),
            hour=self.current_candle.time.hour,
            direction=direction,
            sl=sl_price,
            tp=tp_price,
            entry=self.current_candle.close,
            time=self.current_candle.time,
            open_balance=self.balance
        )

    def exit_trade(self):
        """
        Exit trade.
        """
        res = abs(self.position["exit"] - self.position["entry"]) * 150
        self.position["result"] = (
            res if self.position["win"] else -res
        ) - self.commision  # Commision

        self.balance += self.position["result"]
        self.position["duration"] = self.current_candle.time - self.position["time"]
        self.position["close_balance"] = self.balance
        self.position["trade_return"] = (self.position["open_balance"] - self.balance) / self.position["open_balance"]

        self.trades.append(self.position)
        if self.position["win"]:
            self.win_trades.append(self.position)
        else:
            self.loss_trades.append(self.position)
        self.position = {}

    #  ===========================================================================
    #                               STRATEGY DE TRADING
    #  ===========================================================================
        """_summary_: Describe your different strategies here
        """
    def strategy_BB_MA_Stoch(self):
        last5 = (
            self.current_index - 5 in self.data.index
            and self.get_candle(self.current_index - 5)
            or False
        )
        previous = (
            self.current_index - 1 in self.data.index
            and self.get_candle(self.current_index - 1)
            or False
        )

        bullish_candle = self.is_hammer() or self.is_bullish_engulfing()
        bearish_candle = self.is_star() or self.is_bearish_engulfing()

        historical_candle_above_ema = (
            last5 and last5.open > self.current_candle.ema_high
        )
        historical_candle_below_ema = last5 and last5.open < self.current_candle.ema_low

        candle_touch_ema_from_above = (
            self.current_candle.close > self.current_candle.ema_low
            and self.current_candle.low < self.current_candle.ema_high
        )
        candle_touch_ema_from_below = (
            self.current_candle.close < self.current_candle.ema_high
            and self.current_candle.high > self.current_candle.ema_low
        )

        candle_touch_low_bb = self.current_candle.low < self.current_candle.bb_low or (
            previous and previous.low < self.current_candle.bb_low
        )
        candle_touch_high_bb = (
            self.current_candle.high > self.current_candle.bb_high
            or (previous and previous.high > self.current_candle.bb_high)
        )

        stoch_overbought = self.current_candle.stoch > 65
        stoch_oversold = self.current_candle.stoch < 35

        long_condition = (
            bullish_candle
            and historical_candle_above_ema
            and candle_touch_ema_from_above
            and candle_touch_low_bb
            and stoch_oversold
        )
        short_condition = (
            bearish_candle
            and historical_candle_below_ema
            and candle_touch_ema_from_below
            and candle_touch_high_bb
            and stoch_overbought
        )

        long_condition = True if long_condition else False
        short_condition = True if short_condition else False

        self.data.loc[self.current_index, "long_condition"] = long_condition
        self.data.loc[self.current_index, "short_condition"] = short_condition

        return long_condition, short_condition

    def strategy_ORB_M15(self):
        return False, False

    #  ===========================================================================
    #                               CONDITIONS DE TRADING
    #  ===========================================================================
    def get_condition(self):
        """
        Get the condition for long trades.
        """
        return self.strategy_BB_MA_Stoch()

    #  ===========================================================================
    #                           CANDLESTICKS PATTERNS
    #  ===========================================================================
    def is_bearish_engulfing(self):
        """
        Check if the candlestick is an engulfing pattern.
        """
        previous_candle = (
            self.get_candle(self.current_index - 1)
            if self.current_index - 1 in self.data.index
            else False
        )
        previous_candle_bullish = (
            previous_candle and previous_candle.close >= previous_candle.open
        )
        current_candle_bearish = self.current_candle.close < self.current_candle.open
        current_candle_engulf = (
            self.current_candle.close < previous_candle.open
            and self.current_candle.open >= previous_candle.close
        )
        return (
            previous_candle_bullish and current_candle_bearish and current_candle_engulf
        )

    def is_bullish_engulfing(self):
        """
        Check if the candlestick is an engulfing pattern.
        """
        previous_candle = (
            self.get_candle(self.current_index - 1)
            if self.current_index - 1 in self.data.index
            else False
        )
        previous_candle_bearish = (
            previous_candle and previous_candle.close < previous_candle.open
        )
        current_candle_bullish = self.current_candle.close > self.current_candle.open
        current_candle_engulf = (
            self.current_candle.close > previous_candle.open
            and self.current_candle.open <= previous_candle.close
        )
        return (
            previous_candle_bearish and current_candle_bullish and current_candle_engulf
        )

    def is_hammer(self):
        """
        Check if the candlestick is a hammer pattern.
        """
        lowestBody = (
            self.current_candle.close < self.current_candle.open
            and self.current_candle.close
            or self.current_candle.open
        )
        return lowestBody >= self.current_candle.bull_fib

    def is_star(self):
        """
        Check if the candlestick is a star pattern.
        """
        highest_body = (
            self.current_candle.close > self.current_candle.open
            and self.current_candle.close
            or self.current_candle.open
        )
        return highest_body <= self.current_candle.bear_fib
    
    #  ===========================================================================
    #                                   ANALYSIS
    #  ===========================================================================
    def get_overview(self):
        max_days_drawdown = self.df_trades["drawdown_pct"].max()
        max_drawdown_equity = self.df_trades["drawdown"].max()
        total_pnl = self.balance - self.initial_balance
        overview = [{
            "Total P&L": f"{total_pnl:.2f} Usd",
            "Max equity drawdown": f"{max_drawdown_equity:.2f} Usd  ({max_days_drawdown*100:.2f}%)",
            "Total trades": f"{len(self.trades)}",
            "Profitable trades": self.win_rates
        }]
        self.overview = pd.DataFrame(overview)

    def get_performance(self):
        long_trades = self.df_trades.loc[self.df_trades["direction"] == "buy"]
        short_trades = self.df_trades.loc[self.df_trades["direction"] == "sell"]
        
        net_profit = self.balance - self.initial_balance
        long_net = long_trades["result"].sum()
        short_net = short_trades["result"].sum()
        performance = []
        performance.append({
            "Metric": "Net profit",
            "All": f"{net_profit:.2f} Usd",
            "Long": f"{long_net:.2f} Usd",
            "Short": f"{short_net:.2f} Usd",
        })

        self.gross_profit = self.df_trades[self.df_trades["win"]]["result"].sum()
        long_gross_profit = long_trades[long_trades["win"]]["result"].sum()
        short_gross_profit = short_trades[short_trades["win"]]["result"].sum()
        performance.append({
            "Metric": "Gross profit",
            "All": f"{self.gross_profit:.2f} Usd",
            "Long": f"{long_gross_profit:.2f} Usd",
            "Short": f"{short_gross_profit:.2f} Usd",
        })

        self.gross_loss = self.df_trades[self.df_trades["win"] == False]["result"].sum()
        long_gross_loss = long_trades[long_trades["win"] == False]["result"].sum()
        short_gross_loss = short_trades[short_trades["win"] == False]["result"].sum()
        performance.append({
            "Metric": "Gross loss",
            "All": f"{self.gross_loss:.2f} Usd",
            "Long": f"{long_gross_loss:.2f} Usd",
            "Short": f"{short_gross_loss:.2f} Usd",
        })
        
        total_commission = len(self.trades) * self.commision
        long_commission = len(long_trades) * self.commision
        short_commission = len(short_trades) * self.commision
        performance.append({
            "Metric": "Commission paid",
            "All": f"{total_commission:.2f} Usd",
            "Long": f"{long_commission:.2f} Usd",
            "Short": f"{short_commission:.2f} Usd",
        })

        self.long_profit_factor = abs(long_gross_profit / long_gross_loss)
        self.short_profit_factor = abs(short_gross_profit / short_gross_loss)

        self.performance = pd.DataFrame(performance)

        
    def get_risk_ratio(self):
        rp = self.df_days["daily_return"].mean()
        rf = self.risk_free / 252 # ex: (2% / 252 jours de bourse)
        volatility = self.df_days["daily_return"].std()
        downside_deviation = self.df_days[self.df_days["daily_return"] < 0]["daily_return"].std()

        sharpe_ratio = abs((rp - rf) / volatility)
        sortino_ratio = abs((rp - rf) / downside_deviation)
        self.profit_factor = abs(self.gross_profit / self.gross_loss)

        risk_ratio = [{
            "Metric": "Profit factor",
            "All": f"{self.profit_factor:.3f}",
            "Long": f"{self.long_profit_factor:.3f}",
            "Short": f"{self.short_profit_factor:.3f}",
        }]

        risk_ratio.append({
            "Metric": "Sharp ratio",
            "All": f"{sharpe_ratio:.3f}"
        })

        risk_ratio.append({
            "Metric": "Sortino ratio",
            "All": f"{sortino_ratio:.3f}"
        })


        total_days = self.df_trades['day'].nunique()
        mean_trades_per_days = len(self.trades) / total_days

        risk_ratio.append({
            "Metric": "Avg # trades per day",
            "All": f"{mean_trades_per_days:.2f}"
        })
        self.risk_ratio = pd.DataFrame(risk_ratio)

        
    def get_deep_analysis(self):
        long_trades = self.df_trades.loc[self.df_trades["direction"] == "buy"]
        short_trades = self.df_trades.loc[self.df_trades["direction"] == "sell"]
        
        total_trades = len(self.trades)
        total_long_trades = len(long_trades)
        total_short_trades = len(short_trades)

        analysis = [{
            "Metric": "Total trades",
            "All": f"{total_trades}",
            "Long": f"{total_long_trades}",
            "Short": f"{total_short_trades}",
        }]


        total_winning_trades = len(self.df_trades[self.df_trades["win"]])
        long_winning_trades = len(long_trades[long_trades["win"]])
        short_winning_trades = len(short_trades[short_trades["win"]])
        analysis.append({
            "Metric": "Winning trades",
            "All": f"{total_winning_trades}",
            "Long": f"{long_winning_trades}",
            "Short": f"{short_winning_trades}",
        })

        total_losing_trades = len(self.df_trades[self.df_trades["win"] == False])
        long_losing_trades = len(long_trades[long_trades["win"] == False])
        short_losing_trades = len(short_trades[short_trades["win"] == False])
        analysis.append({
            "Metric": "Losing trades",
            "All": f"{total_losing_trades}",
            "Long": f"{long_losing_trades}",
            "Short": f"{short_losing_trades}",
        })

        percentage_profitable = total_winning_trades / total_trades
        long_percentage_profitable = long_winning_trades / total_long_trades 
        short_percentage_profitable = short_winning_trades / total_short_trades 
        analysis.append({
            "Metric": "Percent profitable",
            "All": f"{percentage_profitable * 100:.2f} %",
            "Long": f"{long_percentage_profitable * 100:.2f} %",
            "Short": f"{short_percentage_profitable * 100:.2f} %",
        })

        avg_pnl = self.df_trades["result"].mean()
        long_avg_pnl = long_trades["result"].mean()
        short_avg_pnl = short_trades["result"].mean()
        analysis.append({
            "Metric": "Avg P&L",
            "All": f"{avg_pnl:.2f} Usd",
            "Long": f"{long_avg_pnl:.2f} Usd",
            "Short": f"{short_avg_pnl:.2f} Usd",
        })

        avg_winning_trades = self.df_trades[self.df_trades["win"]]["result"].mean()
        long_avg_winning_trades = long_trades[long_trades["win"]]["result"].mean()
        short_avg_winning_trades = short_trades[short_trades["win"]]["result"].mean()
        analysis.append({
            "Metric": "Avg winning trade",
            "All": f"{avg_winning_trades:.2f} Usd",
            "Long": f"{long_avg_winning_trades:.2f} Usd",
            "Short": f"{short_avg_winning_trades:.2f} Usd",
        })

        avg_losing_trades = self.df_trades[self.df_trades["win"] == False]["result"].mean()
        long_avg_losing_trades = long_trades[long_trades["win"] == False]["result"].mean()
        short_avg_losing_trades = short_trades[short_trades["win"] == False]["result"].mean()
        analysis.append({
            "Metric": "Avg losing trade",
            "All": f"{avg_losing_trades:.2f} Usd",
            "Long": f"{long_avg_losing_trades:.2f} Usd",
            "Short": f"{short_avg_losing_trades:.2f} Usd",
        })

        ratio_avg_win_loss = abs(avg_winning_trades / avg_losing_trades)
        long_ratio_avg_win_loss = abs(long_avg_winning_trades / long_avg_losing_trades)
        short_ratio_avg_win_loss = abs(short_avg_winning_trades / short_avg_losing_trades)
        analysis.append({
            "Metric": "Ratio avg win / avg loss",
            "All": f"{ratio_avg_win_loss:.3f}",
            "Long": f"{long_ratio_avg_win_loss:.3f}",
            "Short": f"{short_ratio_avg_win_loss:.3f}",
        })

        largest_winning_trades = self.df_trades["result"].max()
        largest_long_winning_trades = long_trades["result"].max()
        largest_short_winning_trades = short_trades["result"].max()
        analysis.append({
            "Metric": "Largest winning trade",
            "All": f"{largest_winning_trades:.2f} Usd",
            "Long": f"{largest_long_winning_trades:.2f} Usd",
            "Short": f"{largest_short_winning_trades:.2f} Usd",
        })

        largest_losing_trades = self.df_trades["result"].min()
        largest_long_losing_trades = long_trades["result"].min()
        largest_short_losing_trades = short_trades["result"].min()
        analysis.append({
            "Metric": "Largest losing trade",
            "All": f"{largest_losing_trades:.2f} Usd",
            "Long": f"{largest_long_losing_trades:.2f} Usd",
            "Short": f"{largest_short_losing_trades:.2f} Usd",
        })

        avg_duration_in_trades = self.df_trades["duration"].mean()
        avg_duration_in_long_trades = long_trades["duration"].mean()
        avg_duration_in_short_trades = short_trades["duration"].mean()
        analysis.append({
            "Metric": "Avg # duration in trades",
            "All": f"{avg_duration_in_trades}",
            "Long": f"{avg_duration_in_long_trades}",
            "Short": f"{avg_duration_in_short_trades}",
        })

        avg_duration_winning_trades = self.df_trades[self.df_trades["win"] == True]["duration"].mean()
        avg_duration_winning_long_trades = long_trades[long_trades["win"] == True]["duration"].mean()
        avg_duration_winning_short_trades = short_trades[short_trades["win"] == True]["duration"].mean()
        analysis.append({
            "Metric": "Avg # duration in winning trades",
            "All": f"{avg_duration_winning_trades}",
            "Long": f"{avg_duration_winning_long_trades}",
            "Short": f"{avg_duration_winning_short_trades}",
        })


        avg_duration_losing_trades = self.df_trades[self.df_trades["win"] == False]["duration"].mean()
        avg_duration_losing_long_trades = long_trades[long_trades["win"] == False]["duration"].mean()
        avg_duration_losing_short_trades = short_trades[short_trades["win"] == False]["duration"].mean()
        analysis.append({
            "Metric": "Avg # duration in losing trades",
            "All": f"{avg_duration_losing_trades}",
            "Long": f"{avg_duration_losing_long_trades}",
            "Short": f"{avg_duration_losing_short_trades}",
        })
        self.deep_analysis = pd.DataFrame(analysis)

    def get_daily_analysis(self):
        monday = self.df_days.loc[self.df_days["weekday"] == "Monday"]
        tuesday = self.df_days.loc[self.df_days["weekday"] == "Tuesday"]
        wednesday = self.df_days.loc[self.df_days["weekday"] == "Wednesday"]
        thursday = self.df_days.loc[self.df_days["weekday"] == "Thursday"]
        friday = self.df_days.loc[self.df_days["weekday"] == "Friday"]
        
        total_days = len(self.df_days)
        total_monday = len(monday)
        total_tuesday = len(tuesday)
        total_wednesday = len(wednesday)
        total_thursday = len(thursday)
        total_friday = len(friday)

        analysis = [{
            "Metric": "Total traded days",
            "All": f"{total_days}",
            "Monday": f"{total_monday}",
            "Tuesday": f"{total_tuesday}",
            "Wednesday": f"{total_wednesday}",
            "Thursday": f"{total_thursday}",
            "Friday": f"{total_friday}",
        }]

        total_winning_days = len(self.df_days[self.df_days["result"] > 0])
        monday_winning_days = len(monday[monday["result"] > 0])
        tuesday_winning_days = len(tuesday[tuesday["result"] > 0])
        wednesday_winning_days = len(wednesday[wednesday["result"] > 0])
        thursday_winning_days = len(thursday[thursday["result"] > 0])
        friday_winning_days = len(friday[friday["result"] > 0])
        analysis.append({
            "Metric": "Number of winning days",
            "All": f"{total_winning_days}",
            "Monday": f"{monday_winning_days}",
            "Tuesday": f"{tuesday_winning_days}",
            "Wednesday": f"{wednesday_winning_days}",
            "Thursday": f"{thursday_winning_days}",
            "Friday": f"{friday_winning_days}",
        })

        total_losing_days = len(self.df_days[self.df_days["result"] < 0])
        monday_losing_trades = len(monday[monday["result"] < 0])
        tuesday_losing_trades = len(tuesday[tuesday["result"] < 0])
        wednesday_losing_trades = len(wednesday[wednesday["result"] < 0])
        thursday_losing_trades = len(thursday[thursday["result"] < 0])
        friday_losing_trades = len(friday[friday["result"] < 0])
        analysis.append({
            "Metric": "Number of losing days",
            "All": f"{total_losing_days}",
            "Monday": f"{monday_losing_trades}",
            "Tuesday": f"{tuesday_losing_trades}",
            "Wednesday": f"{wednesday_losing_trades}",
            "Thursday": f"{thursday_losing_trades}",
            "Friday": f"{friday_losing_trades}",
        })

        percentage_profitable = total_winning_days / total_days if total_days > 0 else 0
        monday_percentage_profitable = monday_winning_days / total_monday if total_monday > 0 else 0
        tuesday_percentage_profitable = tuesday_winning_days / total_tuesday if total_tuesday > 0 else 0
        wednesday_percentage_profitable = wednesday_winning_days / total_wednesday if total_wednesday > 0 else 0
        thursday_percentage_profitable = thursday_winning_days / total_thursday if total_thursday > 0 else 0
        friday_percentage_profitable = friday_winning_days / total_friday if total_friday > 0 else 0
        analysis.append({
            "Metric": "Percent profitable",
            "All": f"{percentage_profitable * 100:.2f} %",
            "Monday": f"{monday_percentage_profitable * 100:.2f} %",
            "Tuesday": f"{tuesday_percentage_profitable * 100:.2f} %",
            "Wednesday": f"{wednesday_percentage_profitable * 100:.2f} %",
            "Thursday": f"{thursday_percentage_profitable * 100:.2f} %",
            "Friday": f"{friday_percentage_profitable * 100:.2f} %",
        })

        pnl = self.df_days["result"].sum()
        monday_pnl = monday["result"].sum()
        tuesday_pnl = tuesday["result"].sum()
        wednesday_pnl = wednesday["result"].sum()
        thursday_pnl = thursday["result"].sum()
        friday_pnl = friday["result"].sum()
        analysis.append({
            "Metric": "Net profit",
            "All": f"{pnl:.2f} Usd",
            "Monday": f"{monday_pnl:.2f} Usd",
            "Tuesday": f"{tuesday_pnl:.2f} Usd",
            "Wednesday": f"{wednesday_pnl:.2f} Usd",
            "Thursday": f"{thursday_pnl:.2f} Usd",
            "Friday": f"{friday_pnl:.2f} Usd",
        })

        gross_profit = self.df_days[self.df_days["result"] > 0]["result"].sum()
        monday_gross_profit = monday[monday["result"] > 0]["result"].sum()
        tuesday_gross_profit = tuesday[tuesday["result"] > 0]["result"].sum()
        wednesday_gross_profit = wednesday[wednesday["result"] > 0]["result"].sum()
        thursday_gross_profit = thursday[thursday["result"] > 0]["result"].sum()
        friday_gross_profit = friday[friday["result"] > 0]["result"].sum()
        analysis.append({
            "Metric": "Gross profit",
            "All": f"{gross_profit:.2f} Usd",
            "Monday": f"{monday_gross_profit:.2f} Usd",
            "Tuesday": f"{tuesday_gross_profit:.2f} Usd",
            "Wednesday": f"{wednesday_gross_profit:.2f} Usd",
            "Thursday": f"{thursday_gross_profit:.2f} Usd",
            "Friday": f"{friday_gross_profit:.2f} Usd",
        })

        gross_loss = self.df_days[self.df_days["result"] < 0]["result"].sum()
        monday_gross_loss = monday[monday["result"] < 0]["result"].sum()
        tuesday_gross_loss = tuesday[tuesday["result"] < 0]["result"].sum()
        wednesday_gross_loss = wednesday[wednesday["result"] < 0]["result"].sum()
        thursday_gross_loss = thursday[thursday["result"] < 0]["result"].sum()
        friday_gross_loss = friday[friday["result"] < 0]["result"].sum()
        analysis.append({
            "Metric": "Gross loss",
            "All": f"{gross_loss:.2f} Usd",
            "Monday": f"{monday_gross_loss:.2f} Usd",
            "Tuesday": f"{tuesday_gross_loss:.2f} Usd",
            "Wednesday": f"{wednesday_gross_loss:.2f} Usd",
            "Thursday": f"{thursday_gross_loss:.2f} Usd",
            "Friday": f"{friday_gross_loss:.2f} Usd",
        })

        avg_pnl = self.df_days["result"].mean()
        monday_avg_pnl = monday["result"].mean()
        tuesday_avg_pnl = tuesday["result"].mean()
        wednesday_avg_pnl = wednesday["result"].mean()
        thursday_avg_pnl = thursday["result"].mean()
        friday_avg_pnl = friday["result"].mean()
        analysis.append({
            "Metric": "Avg P&L",
            "All": f"{avg_pnl:.2f} Usd",
            "Monday": f"{monday_avg_pnl:.2f} Usd",
            "Tuesday": f"{tuesday_avg_pnl:.2f} Usd",
            "Wednesday": f"{wednesday_avg_pnl:.2f} Usd",
            "Thursday": f"{thursday_avg_pnl:.2f} Usd",
            "Friday": f"{friday_avg_pnl:.2f} Usd",
        })

        avg_winning_days = self.df_days[self.df_days["result"] > 0]["result"].mean()
        monday_avg_winning_days = monday[monday["result"] > 0]["result"].mean()
        tuesday_avg_winning_days = tuesday[tuesday["result"] > 0]["result"].mean()
        wednesday_avg_winning_days = wednesday[wednesday["result"] > 0]["result"].mean()
        thursday_avg_winning_days = thursday[thursday["result"] > 0]["result"].mean()
        friday_avg_winning_days = friday[friday["result"] > 0]["result"].mean()
        analysis.append({
            "Metric": "Avg winning day",
            "All": f"{avg_winning_days:.2f} Usd",
            "Monday": f"{monday_avg_winning_days:.2f} Usd",
            "Tuesday": f"{tuesday_avg_winning_days:.2f} Usd",
            "Wednesday": f"{wednesday_avg_winning_days:.2f} Usd",
            "Thursday": f"{thursday_avg_winning_days:.2f} Usd",
            "Friday": f"{friday_avg_winning_days:.2f} Usd",
        })

        avg_losing_days = self.df_days[self.df_days["result"] < 0]["result"].mean()
        monday_avg_losing_days = monday[monday["result"] < 0]["result"].mean()
        tuesday_avg_losing_days = tuesday[tuesday["result"] < 0]["result"].mean()
        wednesday_avg_losing_days = wednesday[wednesday["result"] < 0]["result"].mean()
        thursday_avg_losing_days = thursday[thursday["result"] < 0]["result"].mean()
        friday_avg_losing_days = friday[friday["result"] < 0]["result"].mean()
        analysis.append({
            "Metric": "Avg losing day",
            "All": f"{avg_losing_days:.2f} Usd",
            "Monday": f"{monday_avg_losing_days:.2f} Usd",
            "Tuesday": f"{tuesday_avg_losing_days:.2f} Usd",
            "Wednesday": f"{wednesday_avg_losing_days:.2f} Usd",
            "Thursday": f"{thursday_avg_losing_days:.2f} Usd",
            "Friday": f"{friday_avg_losing_days:.2f} Usd",
        })

        ratio_avg_win_loss = abs(avg_winning_days / avg_losing_days)
        monday_ratio_avg_win_loss = abs(monday_avg_winning_days / monday_avg_losing_days)
        tuesday_ratio_avg_win_loss = abs(tuesday_avg_winning_days / tuesday_avg_losing_days)
        wednesday_ratio_avg_win_loss = abs(wednesday_avg_winning_days / wednesday_avg_losing_days)
        thursday_ratio_avg_win_loss = abs(thursday_avg_winning_days / thursday_avg_losing_days)
        friday_ratio_avg_win_loss = abs(friday_avg_winning_days / friday_avg_losing_days)
        analysis.append({
            "Metric": "Ratio avg win / avg loss",
            "All": f"{ratio_avg_win_loss:.3f}",
            "Monday": f"{monday_ratio_avg_win_loss:.3f}",
            "Tuesday": f"{tuesday_ratio_avg_win_loss:.3f}",
            "Wednesday": f"{wednesday_ratio_avg_win_loss:.3f}",
            "Thursday": f"{thursday_ratio_avg_win_loss:.3f}",
            "Friday": f"{friday_ratio_avg_win_loss:.3f}",
        })

        largest_winning_days = self.df_days["result"].max()
        monday_largest_winning_days = monday["result"].max()
        tuesday_largest_winning_days = tuesday["result"].max()
        wednesday_largest_winning_days = wednesday["result"].max()
        thursday_largest_winning_days = thursday["result"].max()
        friday_largest_winning_days = friday["result"].max()
        analysis.append({
            "Metric": "Largest winning day",
            "All": f"{largest_winning_days:.2f} Usd",
            "Monday": f"{monday_largest_winning_days:.2f} Usd",
            "Tuesday": f"{tuesday_largest_winning_days:.2f} Usd",
            "Wednesday": f"{wednesday_largest_winning_days:.2f} Usd",
            "Thursday": f"{thursday_largest_winning_days:.2f} Usd",
            "Friday": f"{friday_largest_winning_days:.2f} Usd",
        })

        largest_losing_days = self.df_days["result"].min()
        monday_largest_losing_days = monday["result"].min()
        tuesday_largest_losing_days = tuesday["result"].min()
        wednesday_largest_losing_days = wednesday["result"].min()
        thursday_largest_losing_days = thursday["result"].min()
        friday_largest_losing_days = friday["result"].min()
        analysis.append({
            "Metric": "Largest losing day",
            "All": f"{largest_losing_days:.2f} Usd",
            "Monday": f"{monday_largest_losing_days:.2f} Usd",
            "Tuesday": f"{tuesday_largest_losing_days:.2f} Usd",
            "Wednesday": f"{wednesday_largest_losing_days:.2f} Usd",
            "Thursday": f"{thursday_largest_losing_days:.2f} Usd",
            "Friday": f"{friday_largest_losing_days:.2f} Usd",
        })

        self.daily_analysis = pd.DataFrame(analysis)

    def analysis(self):
        self.df_trades = pd.DataFrame(self.trades)
        self.df_trades["balance_ath"] = self.df_trades["close_balance"].cummax()
        self.df_trades["drawdown"] = self.df_trades["balance_ath"] - self.df_trades["close_balance"]
        self.df_trades["drawdown_pct"] = self.df_trades["drawdown"] / self.df_trades["balance_ath"]

        self.df_days = self.df_trades[['day', 'result', 'open_balance', 'close_balance']].groupby('day', as_index=False).sum()
        self.df_days["daily_return"] = (self.df_days["open_balance"] - self.df_days["close_balance"]) / self.df_days["open_balance"]
        self.df_days["weekday"] = self.df_days["day"].apply(lambda d: d.strftime('%A'))

        self.get_overview()
        self.get_performance()
        self.get_risk_ratio()
        self.get_deep_analysis()
        self.get_daily_analysis()
        self.hourly_analysis = self.df_trades[['hour', 'result']].groupby('hour', as_index=False).sum()
        

    def show_chart(self):
        fig, ax_left = plt.subplots(figsize=(15, 20), nrows=4, ncols=1)

        ax_left[0].title.set_text("Profit and Lose curve")
        ax_left[0].plot(self.df_trades['close_balance'], color='royalblue', lw=1)
        ax_left[0].fill_between(self.df_trades['close_balance'].index, self.df_trades['close_balance'], alpha=0.2, color='royalblue')
        ax_left[0].axhline(y=self.df_trades.iloc[0]['close_balance'], color='black', alpha=0.3)
        ax_left[0].legend(['Wallet evolution (equity)'], loc ="upper left")

        ax_left[1].title.set_text("Asset evolution")
        ax_left[1].plot(self.df_trades['entry'], color='sandybrown', lw=1)
        ax_left[1].fill_between(self.df_trades['entry'].index, self.df_trades['entry'], alpha=0.2, color='sandybrown')
        ax_left[1].axhline(y=self.df_trades.iloc[0]['entry'], color='black', alpha=0.3)
        ax_left[1].legend(['Asset evolution'], loc ="upper left")

        ax_left[2].title.set_text("Drawdown curve")
        ax_left[2].plot(-self.df_trades['drawdown_pct']*100, color='indianred', lw=1)
        ax_left[2].fill_between(self.df_trades['drawdown_pct'].index, -self.df_trades['drawdown_pct']*100, alpha=0.2, color='indianred')
        ax_left[2].axhline(y=0, color='black', alpha=0.3)
        ax_left[2].legend(['Drawdown in %'], loc ="lower left")

        ax_right = ax_left[3].twinx()

        ax_left[3].title.set_text("P&L VS Asset (not on the same scale)")
        ax_left[3].plot(self.df_trades['close_balance'], color='royalblue', lw=1)
        ax_right.plot(self.df_trades['entry'], color='sandybrown', lw=1)
        ax_left[3].legend(['Wallet evolution (equity)'], loc ="lower right")
        ax_right.legend(['Asset evolution'], loc ="upper left")

        plt.show()

    #  ===========================================================================
    #                                   RUN
    #  ===========================================================================
    def run(self):
        """
        Run the backtest.
        """
        self.calculate_indicators()

        # Boucle sur les bougies
        for index in range(len(self.data)):
            if index == 0:
                continue

            self.current_candle = self.get_candle(index)
            self.current_index = index

            if not self.position:
                # Vérifier si une nouvelle position doit être ouverte

                session_start, session_end = self.get_session(self.start_session_hour, self.end_session_hour, self.start_session_minute, self.end_session_minute)
                monday = self.current_candle.time.weekday() == 0
                in_session = session_start <= self.current_candle.time <= session_end
                if (not in_session):
                    continue

                # Vérifier les conditions de trading
                long_condition, short_condition = self.get_condition()

                # Ouvrir une position si les conditions sont remplies
                if long_condition:
                    self.take_trade("buy")

                if short_condition:
                    self.take_trade("sell")

            else:
                # Vérifier si la position doit être fermée
                SL = (
                    self.position["direction"] == "buy"
                    and self.current_candle.close <= self.position["sl"]
                ) or (
                    self.position["direction"] == "sell"
                    and self.current_candle.close >= self.position["sl"]
                )

                TP = (
                    self.position["direction"] == "buy"
                    and self.current_candle.close >= self.position["tp"]
                ) or (
                    self.position["direction"] == "sell"
                    and self.current_candle.close <= self.position["tp"]
                )

                if SL:
                    self.position.update(win=False, exit=self.position["sl"])

                elif TP:
                    self.position.update(win=True, exit=self.position["tp"])
                else:
                    continue

                self.exit_trade()
