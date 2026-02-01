"""
API 測試共用 fixtures
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import tempfile
import os

from src.api.models import VideoProcessRequest, VideoProcessResponse
from src.models.prediction_result import PredictionResult, SegmentPrediction


@pytest.fixture
def mock_config():
    """模擬配置字典"""
    return {
        "output": {
            "base_dir": tempfile.gettempdir()
        },
        "skeleton_extraction": {},
        "resgcn": {}
    }


@pytest.fixture
def mock_prediction_result():
    """模擬 PredictionResult"""
    segment_pred = SegmentPrediction(
        segment_id="seg_001",
        predicted_class=2,
        confidence=0.85,
        prob_distribution=[0.05, 0.1, 0.85],
        predicted_age=12.5
    )
    
    return PredictionResult(
        case_id="test_case",
        predicted_age=12.5,
        confidence=0.85,
        prob_distribution=[0.05, 0.1, 0.85],
        segment_predictions=[segment_pred],
        actual_age=12.0,
        age_difference=0.5,
        development_status="正常",
        num_segments=1
    )


@pytest.fixture
def mock_prediction_workflow(mock_prediction_result):
    """模擬 PredictionWorkflow"""
    with patch('src.api.main.PredictionWorkflow') as MockWorkflow:
        instance = MockWorkflow.return_value
        instance.predict_from_videos.return_value = mock_prediction_result
        yield MockWorkflow


@pytest.fixture
def mock_config_manager(mock_config):
    """模擬 ConfigManager"""
    with patch('src.api.main.ConfigManager') as MockConfigManager:
        instance = MockConfigManager.return_value
        instance.config = mock_config
        yield MockConfigManager


@pytest.fixture
def test_client(mock_config_manager):
    """建立測試客戶端（使用 mock 配置）"""
    from src.api.main import app
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_video_request():
    """範例影片處理請求"""
    return {
        "case_id": "test_case_001",
        "videopath": "d:/test_video.mp4",
        "months": 12
    }


@pytest.fixture
def temp_video_file():
    """建立臨時測試影片檔案"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(b'fake video content')
        temp_path = f.name
    yield temp_path
    # 清理
    if os.path.exists(temp_path):
        os.unlink(temp_path)
