#!/usr/bin/env python3
"""
測試簡潔架構的腳本
"""

import numpy as np
from src.core.predictor_factory import build_model

def test_xgboost_model():
    """測試XGBoost模型"""
    print("測試XGBoost模型...")

    config = {
        "model_type": "xgboost",
        "num_classes": 5,
        "features": {
            "feature_type": "gait",
            "gait_dim": 256
        },
        "xgboost": {
            "n_estimators": 10,  # 小的數量用於測試
            "max_depth": 3,
            "learning_rate": 0.1
        }
    }

    # 建立模型
    model = build_model(config)
    print(f"模型建立成功: {model}")

    # 測試預測
    gait_features = np.random.randn(256)
    prediction = model.predict(gait_features)
    print(f"單樣本預測結果形狀: {prediction.shape}")
    print(f"預測結果: {prediction}")

    # 測試批次預測
    batch_features = [np.random.randn(256) for _ in range(3)]
    batch_predictions = model.predict_batch(batch_features)
    print(f"批次預測結果形狀: {batch_predictions.shape}")

    # 測試模型資訊
    info = model.get_model_info()
    print(f"模型資訊: {info['architecture']}")

    print("XGBoost模型測試通過!\n")

def test_baseline_model():
    """測試基準MLP模型"""
    print("測試基準MLP模型...")

    config = {
        "model_type": "baseline",
        "num_classes": 5,
        "metrics_dim": 50,
        "hidden_dims": [64, 32],
        "features": {
            "metrics_dim": 50
        }
    }

    # 建立模型
    model = build_model(config)
    print(f"模型建立成功: {model}")

    # 創建測試metrics數據
    metrics = {
        "body_proportion": pd.DataFrame(np.random.randn(10, 5)),
        "standing_metric": pd.DataFrame(np.random.randn(10, 3))
    }

    # 測試預測
    prediction = model.predict(metrics)
    print(f"預測結果形狀: {prediction.shape}")

    # 測試模型資訊
    info = model.get_model_info()
    print(f"架構: {info['architecture']}")

    print("基準MLP模型測試通過!\n")

if __name__ == "__main__":
    import pandas as pd

    print("開始測試簡潔架構...\n")

    try:
        test_xgboost_model()
        test_baseline_model()
        print("所有測試通過! 🎉")
    except Exception as e:
        print(f"測試失敗: {e}")
        import traceback
        traceback.print_exc()