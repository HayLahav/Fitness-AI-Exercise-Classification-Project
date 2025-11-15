"""
Unit tests for API endpoints
"""

import pytest
from fastapi.testclient import TestClient

from fitness_ai.api.main import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


class TestAPI:
    """Test cases for API endpoints"""

    def test_root_endpoint(self, client):
        """Test root endpoint returns health status"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_list_exercises(self, client):
        """Test exercises list endpoint"""
        response = client.get("/exercises")
        assert response.status_code == 200
        data = response.json()
        assert "exercises" in data
        assert "count" in data
        assert len(data["exercises"]) > 0
        assert data["count"] == len(data["exercises"])

    def test_classify_without_file(self, client):
        """Test classify endpoint without file"""
        response = client.post("/classify")
        assert response.status_code == 422  # Validation error

    def test_api_documentation(self, client):
        """Test that API documentation is accessible"""
        response = client.get("/docs")
        assert response.status_code == 200
