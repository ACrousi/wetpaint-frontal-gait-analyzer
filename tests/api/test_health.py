"""
健康檢查端點測試
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


class TestHealthEndpoint:
    """測試根路徑健康檢查"""

    def test_root_returns_ok(self, test_client):
        """測試根路徑回傳 status ok"""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_root_response_format(self, test_client):
        """測試根路徑響應格式包含必要欄位"""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data

    def test_root_message_content(self, test_client):
        """測試回傳訊息內容"""
        response = test_client.get("/")
        data = response.json()
        assert "API" in data["message"] or "運行" in data["message"]
