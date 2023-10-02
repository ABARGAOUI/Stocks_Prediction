from fastapi.testclient import TestClient
import pytest
from main import app

TEST_DATA_LAGS = [({
        "regression_type": "logistic",
        "ticker": "DJI",
        "start_date": "2009-12-31",
        "end_date": "2017-1-1",
        "test_start_date": "2017-1-2",
        "test_end_date": "2019-11-15",
        "amount": 19899.289063,
        "lags": 3
    })]

TEST_DATA_COSTUM_COLS = [({
        "regression_type": "linear",
        "ticker": "DJI",
        "start_date": "2009-12-31",
        "end_date": "2017-1-1",
        "test_start_date": "2017-1-2",
        "test_end_date": "2019-11-15",
        "amount": 19899.289063,
        "columns": ["mom",'mom1','mom2','mom3']
    })]

@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.parametrize("payload", TEST_DATA_LAGS)
def test_regression(client: TestClient, payload):
    response = client.post("/regressions", json=payload)
    print(response)
    assert response.status_code == 200


@pytest.mark.parametrize("payload", TEST_DATA_COSTUM_COLS)
def test_regression_costum_columns(client: TestClient, payload):
    response = client.post("/regressions", json=payload)
    print(response)
    assert response.status_code == 200
