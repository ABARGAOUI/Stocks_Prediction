from pydantic import BaseModel
from enum import Enum
from typing import List


class RegressionType(Enum):
    LINEAR_REGRESSION = "linear"
    LOGISTIC_REGRESSION = "logistic"


class Input(BaseModel):
    # Regression type either linear or logistic
    regression_type: RegressionType
    # Financial instrument to work with
    ticker: str
    # start_date for data Selection
    start_date: str
    # end_date for data selection
    end_date: str
    # start_date for the testing subset
    test_start_date: str
    # end_date for the testing subset
    test_end_date: str
    # amount used as a notional
    amount: float = 1000
    # number of lag days used in the regression computing
    lags: int = 3
    # Columns to use as features
    features: List[str] = None

