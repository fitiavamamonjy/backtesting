from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class Candle:
    index: int
    open: float
    high: float
    low: float
    close: float
    time: datetime  # sera converti depuis pd.Timestamp
    ema_high: float = 0.0 
    ema_low: float = 0.0
    bb_high: float = 0.0
    bb_low: float = 0.0
    stoch: float = 0.0
    fib_ratio: float = 0.5
    bull_fib: float = field(init=False)
    bear_fib: float = field(init=False)

    def __post_init__(self):
        # Conversion explicite de pandas.Timestamp vers datetime.datetime
        self.time = self.time.to_pydatetime()

        # Calcul des niveaux Fibonacci
        self.bull_fib = (self.low - self.high) * self.fib_ratio + self.high
        self.bear_fib = (self.high - self.low) * self.fib_ratio + self.low

    __repr__ = lambda self: f"Candle(open={self.open}, high={self.high}, low={self.low}, close={self.close}, time={self.time})"
    __str__ = lambda self: f"Candle(open={self.open}, high={self.high}, low={self.low}, close={self.close}, time={self.time})"
    __eq__ = lambda self, other: self.open == other.open and self.high == other.high and self.low == other.low and self.close == other.close
