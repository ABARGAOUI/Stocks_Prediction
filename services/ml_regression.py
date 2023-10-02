import numpy as np
import pandas as pd
from typing import List
from sklearn import linear_model
from models.input import RegressionType
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class ScikitRegression:
    """ Class for predicting index levels using regressions

        Attributes
        ==========
        ticker: str
            ticker to work with
        start: str
            start date of data selection
        end: str
            end date of data selection
        model: RegressionType
            either 'regression' or 'logistic'
        lags: int
            day lags to use as features
        amount: float
            amount to be invested at the beginning, If we want to predict the exact price we enter j-1 price
        columns: List[str]
            columns to use as features to fit the model

        Methods
        =======
        get_data:
            retrieve data from the corresponding CSV file in the data directory
        select_data_sub_set:
            selects a sub-set of data
        prepare_features:
            prepares the feature data for the model fitting
        fit_model:
            implements the fitting step
        compute:
            compute prediction on log returns and the corresponding price performance
        compute__regression_performance:
            compute how-many times the sign of the realized returns is the same as the predicted one
            compute different performance indicators
    """

    def __init__(self, ticker: str, start: str, end: str, model: RegressionType, lag: int, amount: float,
                 columns=None):
        if columns is None:
            self.features_columns = []
            self.use_lags = True
        else:
            self.features_columns = columns
        self.ticker = ticker
        self.start = start
        self.end = end
        self.amount = amount
        self.lags = lag
        self.results, self.data, self.data_subset = None, None, None
        if model == RegressionType.LINEAR_REGRESSION:
            self.model = linear_model.LinearRegression()
        elif model == RegressionType.LOGISTIC_REGRESSION:
            self.model = linear_model.LogisticRegression(C=1e6, solver='lbfgs', multi_class='ovr', max_iter=1000)
        else:
            raise ValueError('Model not yet implemented')
        self.get_data()

    def get_data(self) -> None:
        try:
            raw_data = pd.read_csv(f'data/Processed_{self.ticker}.csv', index_col=0, parse_dates=True).dropna()
            raw_data = pd.DataFrame(raw_data[['Close']+self.features_columns])
        except BaseException as ex:
            raise ValueError(f"An issue occurred when trying to get historical prices from CSV file: {ex}")

        if raw_data.empty:
            raise ValueError("There is no historical data file is empty.")

        raw_data = raw_data.loc[self.start:self.end]
        raw_data['log_returns'] = np.log(raw_data['Close']/raw_data['Close'].shift(1))
        self.data = raw_data.dropna()

    def select_data_sub_set(self, start: str, end: str) -> pd.DataFrame:
        return self.data[(end >= self.data.index) & (self.data.index >= start)].copy()

    def prepare_features(self, start: str, end: str) -> None:
        self.data_subset = self.select_data_sub_set(start, end)
        if self.use_lags:
            for lag in range(1, 1 + self.lags):
                col = f'lag_{lag}'
                self.data_subset[col] = self.data_subset['log_returns'].shift(lag)
                self.features_columns.append(col)
                self.features_columns = list(set(self.features_columns))
        self.data_subset.dropna(inplace=True)

    def fit_model(self, start: str, end: str) -> None:
        self.prepare_features(start, end)
        self.model.fit(self.data_subset[self.features_columns], np.sign(self.data_subset['log_returns']))

    def compute(self, start_in: str, end_in: str, start_out: str, end_out: str) -> pd.DataFrame:
        self.fit_model(start_in, end_in)
        self.prepare_features(start_out, end_out)
        prediction = self.model.predict(self.data_subset[self.features_columns])
        self.data_subset['prediction'] = prediction
        self.data_subset['predicted_log_returns'] = (self.data_subset['prediction'] * self.data_subset['log_returns'])
        self.data_subset['cumulative_returns'] = self.amount * self.data_subset['log_returns'].cumsum().apply(np.exp)
        self.data_subset['predicted_cumulative_returns'] = self.amount * self.data_subset['predicted_log_returns'].cumsum().apply(np.exp)
        self.results = self.data_subset

        return self.data_subset

    def compute_regression_performance(self):
        hits = np.sign(self.results['log_returns'] * self.results['predicted_log_returns']).value_counts()
        realized = self.results['cumulative_returns']
        predicted = self.results['predicted_cumulative_returns']
        return {"hits": str(hits),
                "hits_mean": str(hits.values[0] / sum(hits)),
                "RMSE": np.sqrt(mean_squared_error(realized, predicted)),
                "r2_square": r2_score(realized, predicted),
                "MAE": mean_absolute_error(realized, predicted)}
