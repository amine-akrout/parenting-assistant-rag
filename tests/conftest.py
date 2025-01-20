import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.api.main import app


@pytest.fixture
def client():
    """Create a test client for FastAPI."""
    return TestClient(app)
