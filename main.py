from models.input import Input
from services.ml_regression import ScikitRegression
from fastapi import FastAPI

app = FastAPI()


@app.post('/regressions')
async def compute_regression(input: Input):
    regression = ScikitRegression(input.ticker, input.start_date, input.test_end_date, input.regression_type,
                                  input.lags, input.amount)
    data = regression.compute(input.start_date, input.end_date, input.test_start_date, input.test_end_date)
    l_reg_performance = regression.compute_regression_performance()
    return {"Performance": l_reg_performance,
            "data": data}

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
