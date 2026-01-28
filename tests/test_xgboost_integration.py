#!/usr/bin/env python3
"""
測試XGBoost整合
"""

import numpy as np
from src.core.pipeline_builder import build_model

def test_xgboost_integration():
    """測試XGBoost模型整合"""

    # 測試配置
    config = {
        "model": {
            "type": "xgboost"
        },
        "xgboost": {
            "num_classes": 5,
            "max_depth": 3,
            "n_estimators": 10,
            "learning_rate": 0.1
        },
        "features": {
            "gait_feature_dim": 50
        }
    }

    print("建立XGBoost模型...")
    try:
        model = build_model(config)
        print("✓ 模型建立成功")
    except Exception as e:
        print(f"✗ 模型建立失敗: {e}")
        return False

    # 測試模型資訊
    try:
        info = model.get_model_info()
        print(f"✓ 模型資訊取得成功: {info['model_type']}")
    except Exception as e:
        print(f"✗ 取得模型資訊失敗: {e}")
        return False

    # 測試預測（未訓練模型）
    try:
        # 模擬metrics資料
        mock_metrics = {
            'body_proportion': None,  # 空的metrics
        }

        result = model(mock_metrics)
        print(f"✓ 預測成功，輸出形狀: {result.shape}")
        print(f"  預測結果: {result}")

        # 檢查輸出是否為機率分佈
        if abs(np.sum(result) - 1.0) < 0.1:  # 允許一些誤差
            print("✓ 輸出為有效的機率分佈")
        else:
            print(f"⚠ 輸出總和不為1: {np.sum(result)}")

    except Exception as e:
        print(f"✗ 預測失敗: {e}")
        return False

    print("🎉 XGBoost整合測試完成！")
    return True

if __name__ == "__main__":
    test_xgboost_integration()