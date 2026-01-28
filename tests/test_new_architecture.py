#!/usr/bin/env python3
"""
測試新的簡潔架構：四種模型預測
"""

import logging
import numpy as np
from pathlib import Path

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gait_mlp():
    """測試Gait MLP"""
    from src.prediction.gait_mlp import GaitMLPPredictor

    config = {
        "num_classes": 5,
        "input_dim": 10,
        "hidden_dims": [32, 16],
        "device": "cpu"
    }

    predictor = GaitMLPPredictor(config)

    # 測試單筆
    features = {"gait": np.random.randn(10).astype(np.float32)}
    probs = predictor.predict_batch_from_features(features)
    assert probs.shape == (5,), f"預期 (5,), 得到 {probs.shape}"

    # 測試批次
    features = {"gait": np.random.randn(3, 10).astype(np.float32)}
    probs = predictor.predict_batch_from_features(features)
    assert probs.shape == (3, 5), f"預期 (3, 5), 得到 {probs.shape}"

    logger.info("✓ Gait MLP 測試通過")

def test_gait_xgb():
    """測試Gait XGBoost"""
    from src.prediction.gait_xgb import GaitXGBPredictor

    config = {
        "num_classes": 5,
        "xgboost": {
            "max_depth": 3,
            "n_estimators": 10,
            "random_state": 42
        }
    }

    predictor = GaitXGBPredictor(config)

    # 訓練簡單資料
    X = np.random.randn(20, 10)
    y = np.random.randint(0, 5, 20)
    predictor.classifier.fit(X, y)

    # 測試預測
    features = {"gait": np.random.randn(10).astype(np.float32)}
    probs = predictor.predict_batch_from_features(features)
    assert probs.shape == (5,), f"預期 (5,), 得到 {probs.shape}"

    logger.info("✓ Gait XGBoost 測試通過")

def test_resgcn():
    """測試ResGCN"""
    from src.prediction.resgcn_predictor import ResGCNPredictor

    config = {
        "resgcn": {
            "num_classes": 5,
            "max_frames": 50,
            "gpus": []
        }
    }

    predictor = ResGCNPredictor(config)

    # 測試單筆 (C, T, V)
    skeleton = np.random.randn(3, 50, 17).astype(np.float32)
    features = {"skeleton": skeleton}
    probs = predictor.predict_batch_from_features(features)
    assert probs.shape == (5,), f"預期 (5,), 得到 {probs.shape}"

    logger.info("✓ ResGCN 測試通過")

def test_fusion():
    """測試Fusion"""
    from src.prediction.fusion_predictor import FusionPredictor

    config = {
        "num_classes": 5,
        "fusion_method": "weighted",
        "fusion_weights": [0.5, 0.5],
        "gait_dim": 10,
        "gait_hidden_dims": [32],
        "resgcn": {
            "num_classes": 5,
            "max_frames": 50,
            "gpus": []
        }
    }

    predictor = FusionPredictor(config)

    # 測試融合
    features = {
        "gait": np.random.randn(10).astype(np.float32),
        "skeleton": np.random.randn(3, 50, 17).astype(np.float32)
    }
    probs = predictor.predict_batch_from_features(features)
    assert probs.shape == (5,), f"預期 (5,), 得到 {probs.shape}"

    logger.info("✓ Fusion 測試通過")

def test_factory():
    """測試工廠"""
    from src.core.predictor_factory import build_predictor

    # 測試各型別
    configs = [
        {"model": {"type": "gait_mlp"}, "num_classes": 5, "input_dim": 10},
        {"model": {"type": "gait_xgboost"}, "num_classes": 5, "xgboost": {"n_estimators": 5}},
        {"model": {"type": "resgcn"}, "resgcn": {"num_classes": 5, "max_frames": 50, "gpus": []}},
        {"model": {"type": "fusion"}, "num_classes": 5, "fusion_method": "weighted", "gait_dim": 10, "resgcn": {"num_classes": 5, "max_frames": 50, "gpus": []}}
    ]

    for config in configs:
        predictor = build_predictor(config)
        info = predictor.get_model_info()
        assert "model_type" in info, f"缺少 model_type: {info}"
        logger.info(f"✓ {config['model']['type']} 工廠測試通過")

def test_workflow():
    """測試工作流"""
    from src.core.batch_prediction_workflow import BatchPredictionWorkflow

    # 模擬簡單配置（無實際檔案）
    config = {
        "model": {"type": "gait_mlp"},
        "data": {"metadata_csv_paths": []},  # 空列表測試
        "num_classes": 5,
        "input_dim": 10
    }

    workflow = BatchPredictionWorkflow(config)
    result = workflow.run()

    # 應返回成功但無預測（無資料）
    assert result["success"] == True, "工作流應成功"
    assert result["predictions"] == [], "應無預測結果"
    assert result["model_type"] == "gait_mlp", f"型別錯誤: {result['model_type']}"

    logger.info("✓ Workflow 測試通過")

if __name__ == "__main__":
    logger.info("開始測試新架構...")

    try:
        test_gait_mlp()
        test_gait_xgb()
        test_resgcn()
        test_fusion()
        test_factory()
        test_workflow()

        logger.info("🎉 所有測試通過！新架構正常運作。")

    except Exception as e:
        logger.error(f"測試失敗: {e}")
        raise