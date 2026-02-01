"""
錯誤處理測試
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException

from src.api.models import (
    VideoProcessingError,
    VideoFileNotFoundError,
    InvalidRequestError
)


class TestVideoProcessingError:
    """測試 VideoProcessingError 處理"""

    def test_default_error_code(self):
        """測試預設錯誤碼為 500"""
        error = VideoProcessingError("測試錯誤")
        assert error.error_code == 500

    def test_custom_error_code(self):
        """測試自定義錯誤碼"""
        error = VideoProcessingError("測試錯誤", 400)
        assert error.error_code == 400

    def test_error_message(self):
        """測試錯誤訊息"""
        error = VideoProcessingError("這是錯誤訊息")
        assert error.message == "這是錯誤訊息"
        assert str(error) == "這是錯誤訊息"


class TestVideoFileNotFoundError:
    """測試 VideoFileNotFoundError 處理"""

    def test_error_code_404(self):
        """測試錯誤碼為 404"""
        error = VideoFileNotFoundError("/path/to/video.mp4")
        assert error.error_code == 404

    def test_error_message_contains_path(self):
        """測試錯誤訊息包含檔案路徑"""
        filepath = "/path/to/missing_video.mp4"
        error = VideoFileNotFoundError(filepath)
        assert filepath in error.message

    def test_inherits_from_video_processing_error(self):
        """測試繼承自 VideoProcessingError"""
        error = VideoFileNotFoundError("/path")
        assert isinstance(error, VideoProcessingError)


class TestInvalidRequestError:
    """測試 InvalidRequestError 處理"""

    def test_error_code_400(self):
        """測試錯誤碼為 400"""
        error = InvalidRequestError("無效參數")
        assert error.error_code == 400

    def test_error_message_format(self):
        """測試錯誤訊息格式"""
        error = InvalidRequestError("缺少必要欄位")
        assert "無效請求" in error.message
        assert "缺少必要欄位" in error.message

    def test_inherits_from_video_processing_error(self):
        """測試繼承自 VideoProcessingError"""
        error = InvalidRequestError("msg")
        assert isinstance(error, VideoProcessingError)


class TestErrorInAPIContext:
    """測試 API 上下文中的錯誤處理"""

    def test_file_not_found_returns_404(self, test_client, mock_config_manager):
        """測試檔案不存在在 API 中返回 404"""
        response = test_client.post("/process_video", json={
            "case_id": "test",
            "videopath": "/definitely/not/exists/video.mp4",
            "months": 12
        })
        
        assert response.status_code == 404

    def test_processing_error_returns_custom_code(self, test_client, temp_video_file, mock_prediction_workflow):
        """測試處理錯誤返回自定義錯誤碼"""
        mock_prediction_workflow.return_value.predict_from_videos.side_effect = \
            VideoProcessingError("處理失敗", 503)
        
        response = test_client.post("/process_video", json={
            "case_id": "test",
            "videopath": temp_video_file,
            "months": 12
        })
        
        assert response.status_code == 503

    def test_unexpected_error_returns_500(self, test_client, temp_video_file, mock_prediction_workflow):
        """測試未預期錯誤返回 500"""
        mock_prediction_workflow.return_value.predict_from_videos.side_effect = \
            RuntimeError("未預期的錯誤")
        
        response = test_client.post("/process_video", json={
            "case_id": "test",
            "videopath": temp_video_file,
            "months": 12
        })
        
        assert response.status_code == 500
        assert "錯誤" in response.json()["detail"]

    def test_validation_error_returns_422(self, test_client):
        """測試驗證錯誤返回 422"""
        # 發送無效類型
        response = test_client.post("/process_video", json={
            "case_id": 123,  # 應該是字串，但 pydantic 會強制轉換
            "videopath": "/path",
            "months": "not_a_number"  # 這個會失敗
        })
        
        assert response.status_code == 422

    def test_empty_body_returns_422(self, test_client):
        """測試空請求體返回 422"""
        response = test_client.post(
            "/process_video",
            content="",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
