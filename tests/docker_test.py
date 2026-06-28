import requests

BASE_URL = "http://localhost:8000"


def test_health_endpoint():
    res = requests.get(f"{BASE_URL}/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_predict_endpoint():
    payload = {}  # relies on your BASE_LOAN defaults

    res = requests.post(f"{BASE_URL}/predict", json=payload)

    assert res.status_code == 200
    data = res.json()

    assert "default_probability" in data
    assert "prediction" in data
    assert 0 <= data["default_probability"] <= 1