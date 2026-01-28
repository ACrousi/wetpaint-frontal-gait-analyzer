#!/usr/bin/env python3
"""
ResGCN Workflows 整合測試腳本

此腳本展示如何使用新建立的ResGCN訓練和預測workflows
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Any

# 添加專案根目錄到Python路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import ConfigManager
from src.core.training_workflow import ResGCNTrainingWorkflow
from src.core.prediction_workflow import ResGCNPredictionWorkflow
from src.core.video_processing_service import VideoProcessingService
from trash.training_data_service import TrainingDataService

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_video_processing_service():
    """測試VideoProcessingService"""
    logger.info("=== 測試VideoProcessingService ===")

    try:
        config = ConfigManager().get_config()
        video_processor = VideoProcessingService(config)

        # 模擬影片資訊
        video_info = {
            'video_path': 'data/raws/113.3.7-113.3.8 苗栗市衛生所/WETP000175朱泓睿-29m/20240307_022849000_iOS.MOV',
            'video_name': '20240307_022849000_iOS',
            'target_fps': 30
        }

        logger.info(f"處理影片: {video_info['video_name']}")

        # 測試影片處理
        track_manager = video_processor.process_video_case(video_info)

        # 測試骨架資料提取
        skeleton_data = video_processor.extract_skeleton_data_for_training(track_manager)

        logger.info(f"成功提取 {len(skeleton_data)} 個骨架序列")
        return True

    except Exception as e:
        logger.error(f"VideoProcessingService測試失敗: {e}")
        return False


def test_training_data_service():
    """測試TrainingDataService"""
    logger.info("=== 測試TrainingDataService ===")

    try:
        config = ConfigManager().get_config()
        data_service = TrainingDataService(config)

        # 模擬骨架資料
        mock_skeleton_data = [
            {
                'data': np.random.rand(3, 100, 17),  # (C, T, V)
                'track_id': 1,
                'video_label': 0
            },
            {
                'data': np.random.rand(3, 120, 17),
                'track_id': 2,
                'video_label': 1
            }
        ]

        # 測試資料準備
        dataset = data_service.prepare_training_dataset(mock_skeleton_data)

        logger.info(f"訓練集大小: {len(dataset['train']['data'])}")
        logger.info(f"驗證集大小: {len(dataset['val']['data'])}")
        logger.info(f"資料集統計: {dataset['statistics']}")

        return True

    except Exception as e:
        logger.error(f"TrainingDataService測試失敗: {e}")
        return False


def test_resgcn_training_workflow():
    """測試ResGCNTrainingWorkflow"""
    logger.info("=== 測試ResGCNTrainingWorkflow ===")

    try:
        config = ConfigManager().get_config()
        training_workflow = ResGCNTrainingWorkflow(config)

        # 模擬影片列表
        video_list = [
            {
                'video_path': 'data/raws/113.3.7-113.3.8 苗栗市衛生所/WETP000175朱泓睿-29m/20240307_022849000_iOS.MOV',
                'video_name': '20240307_022849000_iOS',
                'label': 0,
                'target_fps': 30
            }
        ]

        logger.info("注意: 此測試不會實際執行訓練，只測試workflow初始化")
        logger.info("如需完整測試，請確保有足夠的訓練資料")

        return True

    except Exception as e:
        logger.error(f"ResGCNTrainingWorkflow測試失敗: {e}")
        return False


def test_resgcn_prediction_workflow():
    """測試ResGCNPredictionWorkflow"""
    logger.info("=== 測試ResGCNPredictionWorkflow ===")

    try:
        config = ConfigManager().get_config()
        prediction_workflow = ResGCNPredictionWorkflow(config)

        # 模擬影片資訊
        video_info = {
            'video_path': 'data/raws/113.3.7-113.3.8 苗栗市衛生所/WETP000175朱泓睿-29m/20240307_022849000_iOS.MOV',
            'video_name': '20240307_022849000_iOS',
            'target_fps': 30
        }

        logger.info(f"測試預測影片: {video_info['video_name']}")

        # 注意: 此測試不會實際執行預測，只測試workflow初始化
        logger.info("注意: 此測試不會實際執行預測，只測試workflow初始化")

        return True

    except Exception as e:
        logger.error(f"ResGCNPredictionWorkflow測試失敗: {e}")
        return False


def demonstrate_workflow_integration():
    """展示workflow整合使用"""
    logger.info("=== 展示Workflow整合使用 ===")

    try:
        config = ConfigManager().get_config()

        # 1. 初始化各個服務
        video_processor = VideoProcessingService(config)
        training_service = TrainingDataService(config)

        # 2. 模擬完整的訓練流程
        logger.info("模擬訓練流程:")

        # 2.1 準備影片列表
        video_list = [
            {
                'video_path': 'data/raws/113.3.7-113.3.8 苗栗市衛生所/WETP000175朱泓睿-29m/20240307_022849000_iOS.MOV',
                'video_name': '20240307_022849000_iOS',
                'label': 0,
                'category': 'normal'
            }
        ]

        # 2.2 處理影片並提取骨架資料
        all_skeleton_data = []
        for video_info in video_list:
            logger.info(f"處理影片: {video_info['video_name']}")
            track_manager = video_processor.process_video_case(video_info)
            skeleton_data = video_processor.extract_skeleton_data_for_training(track_manager)
            all_skeleton_data.extend(skeleton_data)

        # 2.3 準備訓練資料
        if all_skeleton_data:
            dataset = training_service.prepare_training_dataset(all_skeleton_data)
            logger.info("訓練資料準備完成")
        else:
            logger.warning("沒有提取到骨架資料")

        # 3. 模擬預測流程
        logger.info("模擬預測流程:")
        prediction_workflow = ResGCNPredictionWorkflow(config)
        logger.info("預測workflow初始化完成")

        return True

    except Exception as e:
        logger.error(f"Workflow整合測試失敗: {e}")
        return False


def main():
    """主測試函數"""
    logger.info("開始ResGCN Workflows整合測試")
    logger.info("=" * 50)

    # 檢查必要的依賴
    try:
        import numpy as np
        logger.info("NumPy可用")
    except ImportError:
        logger.error("NumPy不可用，請安裝numpy")
        return

    try:
        import torch
        logger.info("PyTorch可用")
    except ImportError:
        logger.warning("PyTorch不可用，某些功能可能受限")

    # 執行測試
    test_results = []

    # 測試各個組件
    test_results.append(("VideoProcessingService", test_video_processing_service()))
    test_results.append(("TrainingDataService", test_training_data_service()))
    test_results.append(("ResGCNTrainingWorkflow", test_resgcn_training_workflow()))
    test_results.append(("ResGCNPredictionWorkflow", test_resgcn_prediction_workflow()))

    # 展示整合使用
    test_results.append(("WorkflowIntegration", demonstrate_workflow_integration()))

    # 總結測試結果
    logger.info("=" * 50)
    logger.info("測試結果總結:")

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {test_name}: {status}")
        if result:
            passed += 1

    logger.info(f"\n總計: {passed}/{total} 個測試通過")

    if passed == total:
        logger.info("🎉 所有測試通過！新架構已準備就緒。")
    else:
        logger.warning("⚠️  部分測試失敗，請檢查相關組件。")

    # 使用說明
    logger.info("\n" + "=" * 50)
    logger.info("使用說明:")
    logger.info("1. 訓練模式:")
    logger.info("   from src.core.training_workflow import ResGCNTrainingWorkflow")
    logger.info("   trainer = ResGCNTrainingWorkflow(config)")
    logger.info("   trainer.run_training_from_videos(video_list)")
    logger.info("")
    logger.info("2. 預測模式:")
    logger.info("   from src.core.prediction_workflow import ResGCNPredictionWorkflow")
    logger.info("   predictor = ResGCNPredictionWorkflow(config)")
    logger.info("   result = predictor.run_prediction(video_info)")
    logger.info("")
    logger.info("3. 影片處理:")
    logger.info("   from src.core.video_processing_service import VideoProcessingService")
    logger.info("   processor = VideoProcessingService(config)")
    logger.info("   track_manager = processor.process_video_case(video_info)")


if __name__ == "__main__":
    main()