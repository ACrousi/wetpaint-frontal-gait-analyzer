"""
/process_video 端點測試
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import os

from src.api.models import VideoFileNotFoundError, VideoProcessingError
from src.models.prediction_result import PredictionResult, SegmentPrediction


class TestProcessVideoEndpoint:
    """測試 /process_video 端點"""

    def test_process_video_success(self, test_client, temp_video_file, mock_prediction_workflow, mock_prediction_result):
        """測試成功處理影片"""
        # 設置 mock
        mock_prediction_workflow.return_value.predict_from_videos.return_value = mock_prediction_result
        
        request_data = {
            "case_id": "test_case_001",
            "videopath": temp_video_file,
            "months": 12
        }
        
        response = test_client.post("/process_video", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["case_id"] == "test_case_001"
        assert "predicted_age" in data
        assert "confidence" in data
        assert "timestamp" in data

    def test_process_video_response_fields(self, test_client, temp_video_file, mock_prediction_workflow, mock_prediction_result):
        """測試響應包含所有必要欄位"""
        mock_prediction_workflow.return_value.predict_from_videos.return_value = mock_prediction_result
        
        request_data = {
            "case_id": "test_case",
            "videopath": temp_video_file,
            "months": 18
        }
        
        response = test_client.post("/process_video", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        required_fields = [
            "case_id", "predicted_age", "predicted_class", 
            "confidence", "prob_distribution", "development_status",
            "num_segments", "timestamp"
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_process_video_file_not_found(self, test_client, mock_config_manager):
        """測試檔案不存在返回 404"""
        request_data = {
            "case_id": "test_case",
            "videopath": "/nonexistent/path/video.mp4",
            "months": 12
        }
        
        response = test_client.post("/process_video", json=request_data)
        
        assert response.status_code == 404
        assert "不存在" in response.json()["detail"] or "not found" in response.json()["detail"].lower()

    def test_process_video_missing_case_id(self, test_client):
        """測試缺少 case_id 返回 422"""
        request_data = {
            "videopath": "/path/to/video.mp4",
            "months": 12
        }
        
        response = test_client.post("/process_video", json=request_data)
        
        assert response.status_code == 422

    def test_process_video_missing_videopath(self, test_client):
        """測試缺少 videopath 返回 422"""
        request_data = {
            "case_id": "test_case",
            "months": 12
        }
        
        response = test_client.post("/process_video", json=request_data)
        
        assert response.status_code == 422

    def test_process_video_missing_months(self, test_client):
        """測試缺少 months 返回 422"""
        request_data = {
            "case_id": "test_case",
            "videopath": "/path/to/video.mp4"
        }
        
        response = test_client.post("/process_video", json=request_data)
        
        assert response.status_code == 422

    def test_process_video_invalid_months_type(self, test_client):
        """測試 months 類型錯誤返回 422"""
        request_data = {
            "case_id": "test_case",
            "videopath": "/path/to/video.mp4",
            "months": "twelve"
        }
        
        response = test_client.post("/process_video", json=request_data)
        
        assert response.status_code == 422

    def test_process_video_internal_error(self, test_client, temp_video_file, mock_prediction_workflow):
        """測試內部錯誤返回 500"""
        mock_prediction_workflow.return_value.predict_from_videos.side_effect = Exception("模擬內部錯誤")
        
        request_data = {
            "case_id": "test_case",
            "videopath": temp_video_file,
            "months": 12
        }
        
        response = test_client.post("/process_video", json=request_data)
        
        assert response.status_code == 500
        assert "錯誤" in response.json()["detail"] or "error" in response.json()["detail"].lower()

    def test_process_video_empty_request(self, test_client):
        """測試空請求返回 422"""
        response = test_client.post("/process_video", json={})
        
        assert response.status_code == 422

    def test_process_video_wrong_method(self, test_client):
        """測試使用 GET 方法返回 405"""
        response = test_client.get("/process_video")
        
        assert response.status_code == 405
