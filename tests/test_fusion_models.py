#!/usr/bin/env python3
"""
測試融合模型和baseline模型的整合
"""

import sys
import os
import logging
from pathlib import Path

# 添加src目錄到路徑
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.config import load_config
from core.fusion_prediction_workflow import FusionPredictionWorkflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_fusion_model():
    """測試融合模型"""
    logger.info("開始測試融合模型...")

    # 載入配置
    config = load_config("config/config.yaml")

    # 設定為融合模型
    config["model_type"] = "fusion"

    # 建立workflow
    workflow = FusionPredictionWorkflow(config)

    # 測試模型資訊
    model_info = workflow.get_model_info()
    logger.info(f"融合模型資訊: {model_info}")

    logger.info("融合模型測試完成")


def test_baseline_model():
    """測試baseline模型"""
    logger.info("開始測試baseline模型...")

    # 載入配置
    config = load_config("config/config.yaml")

    # 設定為baseline模型
    config["model_type"] = "baseline"

    # 建立workflow
    workflow = FusionPredictionWorkflow(config)

    # 測試模型資訊
    model_info = workflow.get_model_info()
    logger.info(f"Baseline模型資訊: {model_info}")

    logger.info("Baseline模型測試完成")


def test_model_loading():
    """測試模型載入"""
    logger.info("測試模型載入...")

    try:
        from src.pose_extract.wrapper import FusionWrapper, MLPBaselineWrapper

        # 測試FusionWrapper
        config = load_config("config/config.yaml")
        fusion_config = config.get("fusion", {})
        fusion_wrapper = FusionWrapper(fusion_config)
        logger.info("FusionWrapper載入成功")

        # 測試MLPBaselineWrapper
        baseline_config = config.get("baseline", {})
        baseline_wrapper = MLPBaselineWrapper(baseline_config)
        logger.info("MLPBaselineWrapper載入成功")

        # 取得模型資訊
        fusion_info = fusion_wrapper.get_model_info()
        baseline_info = baseline_wrapper.get_model_info()

        logger.info(f"Fusion模型資訊: {fusion_info}")
        logger.info(f"Baseline模型資訊: {baseline_info}")

        # 清理資源
        fusion_wrapper.close()
        baseline_wrapper.close()

        logger.info("模型載入測試通過")

    except Exception as e:
        logger.error(f"模型載入測試失敗: {e}")
        raise


if __name__ == "__main__":
    try:
        # 測試模型載入
        test_model_loading()

        # 測試融合模型workflow
        test_fusion_model()

        # 測試baseline模型workflow
        test_baseline_model()

        logger.info("所有測試通過！")

    except Exception as e:
        logger.error(f"測試失敗: {e}")
        sys.exit(1)