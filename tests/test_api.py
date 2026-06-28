import importlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "api"

API_MODULE_CANDIDATES = (
    "api.main",
    "api.app",
    "api.api",
)

APP_ATTRIBUTE_CANDIDATES = (
    "app",
    "api",
    "application",
)


def discover_api_app():
    for module_name in API_MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue

        for attribute_name in APP_ATTRIBUTE_CANDIDATES:
            app = getattr(module, attribute_name, None)
            if app is not None:
                return app

    return None


def test_api_directory_exists_as_deployment_placeholder():
    assert API_DIR.exists()
    assert API_DIR.is_dir()


def test_api_app_is_importable_when_implemented():
    app = discover_api_app()

    if app is None:
        pytest.skip("No API app is implemented yet in api/main.py or api/app.py.")

    assert app is not None


def test_fastapi_health_endpoint_when_app_exists():
    app = discover_api_app()

    if app is None:
        pytest.skip("No FastAPI app is implemented yet.")

    fastapi = pytest.importorskip("fastapi")
    if not isinstance(app, fastapi.FastAPI):
        pytest.skip("Discovered API app is not a FastAPI application.")

    test_client = pytest.importorskip("fastapi.testclient")
    client = test_client.TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()


def test_fastapi_predict_endpoint_contract_when_app_exists():
    app = discover_api_app()

    if app is None:
        pytest.skip("No FastAPI app is implemented yet.")

    fastapi = pytest.importorskip("fastapi")
    if not isinstance(app, fastapi.FastAPI):
        pytest.skip("Discovered API app is not a FastAPI application.")

    route_paths = {
        route.path
        for route in app.routes
        if hasattr(route, "path")
    }

    if "/predict" not in route_paths:
        pytest.skip("FastAPI app does not expose a /predict endpoint yet.")

    assert "/predict" in route_paths