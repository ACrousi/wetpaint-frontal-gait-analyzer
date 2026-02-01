"""
Pydantic 模型驗證測試
"""
import pytest
from pydantic import ValidationError

from src.api.models import (
    VideoProcessRequest, 
    VideoProcessResponse,
    VideoProcessingError,
    VideoFileNotFoundError,
    InvalidRequestError
)


class TestVideoProcessRequest:
    """測試影片處理請求模型"""

    def test_valid_request(self):
        """測試有效請求"""
        request = VideoProcessRequest(
            case_id="test_001",
            videopath="/path/to/video.mp4",
            months=12
        )
        assert request.case_id == "test_001"
        assert request.videopath == "/path/to/video.mp4"
        assert request.months == 12

    def test_missing_case_id(self):
        """測試缺少 case_id"""
        with pytest.raises(ValidationError) as exc_info:
            VideoProcessRequest(
                videopath="/path/to/video.mp4",
                months=12
            )
        assert "case_id" in str(exc_info.value)

    def test_missing_videopath(self):
        """測試缺少 videopath"""
        with pytest.raises(ValidationError) as exc_info:
            VideoProcessRequest(
                case_id="test_001",
                months=12
            )
        assert "videopath" in str(exc_info.value)

    def test_missing_months(self):
        """測試缺少 months"""
        with pytest.raises(ValidationError) as exc_info:
            VideoProcessRequest(
                case_id="test_001",
                videopath="/path/to/video.mp4"
            )
        assert "months" in str(exc_info.value)

    def test_invalid_months_type(self):
        """測試 months 類型錯誤"""
        with pytest.raises(ValidationError):
            VideoProcessRequest(
                case_id="test_001",
                videopath="/path/to/video.mp4",
                months="twelve"  # 應該是 int
            )


class TestVideoProcessResponse:
    """測試影片處理響應模型"""

    def test_valid_response(self):
        """測試有效響應"""
        response = VideoProcessResponse(
            case_id="test_001",
            predicted_age=12.5,
            predicted_class=2,
            confidence=0.85,
            prob_distribution=[0.05, 0.1, 0.85],
            actual_age=12.0,
            age_difference=0.5,
            development_status="正常",
            num_segments=1,
            timestamp="2026-02-01T12:00:00Z"
        )
        assert response.case_id == "test_001"
        assert response.predicted_age == 12.5
        assert response.confidence == 0.85

    def test_optional_fields_default(self):
        """測試可選欄位預設值"""
        response = VideoProcessResponse(
            case_id="test_001",
            predicted_age=12.5,
            predicted_class=2,
            confidence=0.85,
            prob_distribution=[0.05, 0.1, 0.85],
            timestamp="2026-02-01T12:00:00Z"
        )
        assert response.actual_age is None
        assert response.age_difference is None
        assert response.development_status == "未知"
        assert response.num_segments == 1

    def test_prob_distribution_list(self):
        """測試機率分布為列表"""
        response = VideoProcessResponse(
            case_id="test_001",
            predicted_age=12.5,
            predicted_class=2,
            confidence=0.85,
            prob_distribution=[0.1, 0.2, 0.3, 0.4],
            timestamp="2026-02-01T12:00:00Z"
        )
        assert len(response.prob_distribution) == 4
        assert isinstance(response.prob_distribution, list)


class TestErrorClasses:
    """測試錯誤類別"""

    def test_video_processing_error(self):
        """測試 VideoProcessingError"""
        error = VideoProcessingError("處理失敗", 500)
        assert error.message == "處理失敗"
        assert error.error_code == 500
        assert str(error) == "處理失敗"

    def test_video_file_not_found_error(self):
        """測試 VideoFileNotFoundError"""
        error = VideoFileNotFoundError("/path/to/missing.mp4")
        assert "影片檔案不存在" in error.message
        assert "/path/to/missing.mp4" in error.message
        assert error.error_code == 404

    def test_invalid_request_error(self):
        """測試 InvalidRequestError"""
        error = InvalidRequestError("參數錯誤")
        assert "無效請求" in error.message
        assert "參數錯誤" in error.message
        assert error.error_code == 400

    def test_error_inheritance(self):
        """測試錯誤繼承關係"""
        file_error = VideoFileNotFoundError("/path")
        invalid_error = InvalidRequestError("msg")
        
        assert isinstance(file_error, VideoProcessingError)
        assert isinstance(invalid_error, VideoProcessingError)
        assert isinstance(file_error, Exception)
