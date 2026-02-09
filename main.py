import logging
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

# 設定模組路徑
# sys.path.append(os.path.join(os.path.dirname(__file__), '/src'))

# 導入配置和日志模組
from src.core.config import ConfigManager
from src.utils.logger_config import setup_logging
from src.utils.metadataManager import MetadataManager

# 導入例外類
from src.exceptions import (
    ConfigLoadError, 
    SubprocessError,
    DataGenerationError,
    PredictionError,
    InvalidInputError,
    MetadataError,
    VideoNotFoundError,
    ArgumentError,
    ConfigValidationError,
)

# 實現metadata紀錄資料 多筆影片預測訓練

def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='影片處理和機器學習工作流程')
    parser.add_argument('--mode', choices=['train', 'predict', 'extract_from_videos'],
                        default='predict', help='執行模式 (預設: predict)')
    parser.add_argument('--config', default='config/config.yaml',
                        help='配置文件路徑 (預設: config/config.yaml)')
    parser.add_argument('--model', choices=['gcn', 'xgboost'], default=None,
                        help='選擇預測/訓練模型: gcn 或 xgboost')
    parser.add_argument('--videos', nargs='+', type=str,
                        help='預測模式的影片路徑列表')
    parser.add_argument('--case_id', type=str, default=None,
                        help='個案 ID（用於預測模式）')
    parser.add_argument('--actual_age', type=float, default=None,
                        help='實際月齡（用於評估預測結果）')
    parser.add_argument('--skip_extraction', action='store_true',
                        help='跳過骨架提取（假設已有 JSON）')
    parser.add_argument('--save_json', action='store_true',
                        help='儲存 JSON 檔案（骨架和片段），預設不儲存使用臨時檔案')

    return parser.parse_args()

def main():
    """主程式入口點 - 重構後版本

    使用新的 VideoProcessingWorkflow 架構進行影片處理。
    """
    # 解析命令行參數
    args = parse_arguments()
    mode = args.mode
    config_path = args.config

    logging.info(f"執行模式: {mode}")
    logging.info(f"配置文件: {config_path}")

    try:
        # 初始化配置管理器
        config_manager = ConfigManager(config_path)
        config = config_manager.config
        
        # 根據模式選擇工作流程
        if mode == "train":
            train_config = config.get("train", {})
            
            if args.model == 'gcn':
                # 直接呼叫 ResGCNv1/main.py
                import subprocess
                
                resgcn_config = train_config.get("resgcn", {})
                resgcn_config_path = resgcn_config.get("config_path", "./config/resgcn_coco_2.yaml")
                
                # Step 1: 生成訓練資料
                logging.info("Step 1: 生成訓練資料...")
                logging.info(f"使用配置: {resgcn_config_path}")
                
                gen_result = subprocess.run(
                    [sys.executable, "-u", "main.py", "--config", resgcn_config_path, "--generate_data"],
                    cwd="vendor/ResGCNv1",
                    env={**os.environ, "PYTHONUNBUFFERED": "1"}
                )
                
                if gen_result.returncode != 0:
                    raise DataGenerationError(
                        f"ResGCN 資料生成失敗",
                        details={"returncode": gen_result.returncode, "config": resgcn_config_path}
                    )
                
                logging.info("資料生成完成")
                
                # Step 2: 執行訓練
                logging.info("Step 2: 開始 ResGCN 訓練...")
                
                train_result = subprocess.run(
                    [sys.executable, "-u", "main.py", "--config", resgcn_config_path],
                    cwd="vendor/ResGCNv1",
                    env={**os.environ, "PYTHONUNBUFFERED": "1"}
                )
                
                if train_result.returncode == 0:
                    logging.info("ResGCN 訓練完成")
                else:
                    raise SubprocessError(
                        f"ResGCN 訓練失敗",
                        details={"returncode": train_result.returncode, "config": resgcn_config_path}
                    )
                    
            elif args.model == 'xgboost':
                from src.models.xgboost_wrapper import XGBoostWrapper
                logging.info("啟動XGBoost訓練模式")
                
                xgboost_config = train_config.get("xgboost", {})
                
                wrapper = XGBoostWrapper(xgboost_config)
                wrapper.generate_train_data()
                
                n_splits = xgboost_config.get("n_splits", 5)
                wrapper.train(n_splits=n_splits)
                logging.info("XGBoost訓練完成")
            else:
                raise ArgumentError(
                    "訓練模式需要指定 --model 參數",
                    details={"valid_options": ["gcn", "xgboost"]}
                )

        elif mode == "predict":
            if args.model != 'gcn':
                raise ArgumentError(
                    "預測模式目前只支援 --model gcn",
                    details={"provided_model": args.model}
                )
            
            if not args.videos:
                raise ArgumentError(
                    "預測模式需要指定 --videos 參數",
                    details={"required_arg": "--videos"}
                )
            
            # 驗證影片路徑
            valid_videos = []
            for v in args.videos:
                if Path(v).exists():
                    valid_videos.append(v)
                else:
                    logging.warning(f"影片不存在，跳過: {v}")
            
            if not valid_videos:
                raise VideoNotFoundError(
                    "沒有有效的影片路徑",
                    details={"provided_videos": args.videos}
                )
            
            from src.core.workflows.prediction_workflow import PredictionWorkflow
            
            logging.info(f"啟動預測模式，共 {len(valid_videos)} 支影片")
            
            # 建立預測工作流程 - 傳遞 workspace_root
            prediction_workflow = PredictionWorkflow(config, workspace_root=config_manager.workspace_root)
            
            try:
                result = prediction_workflow.predict_from_videos(
                    video_paths=valid_videos,
                    case_id=args.case_id,
                    actual_age=args.actual_age,
                    skip_extraction=args.skip_extraction,
                    save_json=args.save_json
                )
                
                # 輸出結果
                logging.info("=" * 50)
                logging.info("預測結果:")
                logging.info(f"  期望月齡: {result.predicted_age:.2f} 月")
                logging.info(f"  信心度: {result.confidence:.3f}")
                logging.info(f"  片段數: {result.num_segments}")
                
                if result.actual_age is not None:
                    logging.info(f"  實際月齡: {result.actual_age:.1f} 月")
                    logging.info(f"  差異: {result.age_difference:.2f} 月")
                    logging.info(f"  發展評估: {result.development_status}")
                
                logging.info("=" * 50)
                
                # 詳細片段結果
                for seg in result.segment_predictions:
                    logging.info(f"  片段 {seg.segment_id}: 預測={seg.predicted_age:.2f}月, 信心度={seg.confidence:.3f}")
                
            except PredictionError as e:
                logging.error(f"預測失敗: {e}")
                raise
            except Exception as e:
                logging.error(f"預測過程發生未預期錯誤: {e}")
                raise PredictionError(f"預測失敗: {e}")
            
            # predict 模式完成，直接返回
            return

        elif mode == "extract_from_videos":
            skeleton_extraction_config = config.get("skeleton_extraction", {})
            metadata_manager = MetadataManager(skeleton_extraction_config.get("metadata", {}))
            if metadata_manager.df.empty:
                raise MetadataError(
                    "沒有找到可處理的影片記錄",
                    details={"reason": "CSV 為空"}
                )
            logging.info(f"找到 {len(metadata_manager.df)} 筆影片記錄待處理")

            # 影片處理模式 - 直接建立 Workflow
            logging.info("啟動影片處理模式")
            
            from src.core.workflows.skeleton_extraction_workflow import SkeletonExtractionWorkflow
            workflow = SkeletonExtractionWorkflow(
                skeleton_extraction_config, 
                workspace_root=config_manager.workspace_root
            )

            # 處理所有影片
            total_videos = len(metadata_manager.df)
            successful_count = 0
            failed_count = 0

            for idx in range(total_videos):
                video_info = metadata_manager.get_video_info(idx)
                if not video_info:
                    logging.warning(f"跳過索引 {idx}：無效的影片資訊")
                    failed_count += 1
                    continue

                video_path = video_info.get('video_path', video_info.get('original_video', ''))
                video_name = video_info.get('video_name', '')

                if not video_path or not Path(video_path).exists():
                    logging.warning(f"跳過 {video_name}：影片檔案不存在 - {video_path}")
                    failed_count += 1
                    continue

                logging.info(f"開始影片處理 ({idx + 1}/{total_videos}): {video_name}")

                # 使用影片處理工作流程處理影片
                result = workflow.extract_analyze_and_export(video_info)

                if result['success']:
                    logging.info(f"成功處理影片: {video_name}")
                    successful_count += 1
                else:
                    logging.error(f"處理影片失敗: {video_name} - {result.get('error', '未知錯誤')}")
                    failed_count += 1

        else:
            raise ArgumentError(
                f"不支援的執行模式: {mode}",
                details={"valid_modes": ["train", "predict", "extract_from_videos"]}
            )

        # 處理結果摘要
        logging.info("=" * 50)
        logging.info("處理完成摘要:")
        if mode == "video_processing":
            # logging.info(f"總個案數: {total_cases}")
            logging.info(f"總影片數: {total_videos}")
            # summary_total_count = total_cases
            summary_label = "影片"
        else:
            summary_total_count = total_videos
            summary_label = "總影片數"
        logging.info(f"成功處理: {successful_count} {summary_label}")
        logging.info(f"處理失敗: {failed_count} {summary_label}")
        logging.info("=" * 50)

    except (ConfigLoadError, ConfigValidationError) as e:
        logging.error(f"配置錯誤: {e}")
        sys.exit(1)
    except ArgumentError as e:
        logging.error(f"參數錯誤: {e}")
        sys.exit(1)
    except (SubprocessError, DataGenerationError) as e:
        logging.error(f"訓練錯誤: {e}")
        sys.exit(1)
    except PredictionError as e:
        logging.error(f"預測錯誤: {e}")
        sys.exit(1)
    except (MetadataError, VideoNotFoundError) as e:
        logging.error(f"資料錯誤: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"主程式執行過程中發生未預期錯誤: {e}", exc_info=True)
        sys.exit(1)


def _generate_processing_report(workflow, output_base, successful_count, failed_count):
    """生成處理報告"""
    try:
        report_path = output_base / "processing_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("影片處理報告\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"成功處理: {successful_count} 筆\n")
            f.write(f"處理失敗: {failed_count} 筆\n")
        
        logging.info(f"處理報告已生成: {report_path}")
        
    except Exception as e:
        logging.warning(f"生成處理報告時發生錯誤: {e}")

def setup_environment():
    """設定執行環境"""
    # 設定日志
    setup_logging()
    
    # 檢查必要的目錄和檔案
    current_dir = Path.cwd()
    config_path = current_dir / "config" / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError("找不到配置檔案 config.yaml")
    
    logging.info(f"使用配置檔案: {config_path}")
    return str(config_path)


if __name__ == "__main__":
    try:
        # 設定執行環境
        config_path = setup_environment()
        
        # 執行主程式
        main()
        
    except KeyboardInterrupt:
        logging.info("程式被用戶中斷")
        sys.exit(0)
    except Exception as e:
        logging.error(f"程式啟動失敗: {e}", exc_info=True)
        sys.exit(1)