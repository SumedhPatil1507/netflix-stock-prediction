import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def arima_forecast(series: pd.Series, steps: int = 30) -> pd.Series:
    """
    Fit ARIMA(5,1,0) on a price series and return a forecast.
    Uses monthly resampling to reduce noise and speed up fitting.
    """
    monthly = series.resample('ME').mean().dropna()
    model   = ARIMA(monthly, order=(5, 1, 0))
    fitted  = model.fit()
    fc      = fitted.get_forecast(steps)
    return fc.predicted_mean
