import json
import logging
import yaml
import os
import sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F

# 把 vendor/ResGCNv1/src 加到 sys.path，確保能找到 model, dataset.graph
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESGCN_SRC = PROJECT_ROOT / 'vendor' / 'ResGCNv1' / 'src'
if str(RESGCN_SRC) not in sys.path:
    sys.path.insert(0, str(RESGCN_SRC))

import model as resgcn_model
from dataset.graph import Graph

logger = logging.getLogger(__name__)


class PathResolver:
    """路徑解析和適配器"""

    @staticmethod
    def resolve_project_paths(config: dict, project_root: Path) -> dict:
        """將相對路徑解析為絕對路徑"""
        resolved = config.copy()

        # 處理 dataset_args.coco 路徑
        if 'dataset_args' in resolved and 'coco' in resolved['dataset_args']:
            coco_config = resolved['dataset_args']['coco']

            # 定義需要解析的路徑鍵
            path_keys = ['path', 'metadata_path', 'data_path', 'label_path',
                        'eval_data_path', 'eval_label_path']

            # 使用迴圈解析所有路徑
            for key in path_keys:
                if key in coco_config and coco_config[key].startswith('../'):
                    coco_config[key] = str(project_root / coco_config[key].replace('../', ''))

        return resolved


class ModelConfigManager:
    """統一的模型配置管理器"""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    def get_resgcn_config(self, mode: str = 'generate') -> dict:
        """獲取適配後的ResGCN配置"""
        # 載入基礎配置
        config_path = self.project_root / 'vendor' / 'ResGCNv1' / 'configs' / 'resgcn_coco.yaml'

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 應用路徑解析
        return PathResolver.resolve_project_paths(config, self.project_root)

    def create_temp_config(self, config: dict) -> str:
        """創建臨時配置文件"""
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False,
                                       dir=self.project_root / 'vendor' / 'ResGCNv1' / 'configs') as f:
            yaml.dump(config, f)
            return f.name


class ResGCNWrapper:
    """ResGCN模型封裝器，用於動作識別預測"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化ResGCNWrapper

        Args:
            config: ResGCN配置字典
        """
        self.config = config
        self.config_manager = ModelConfigManager(Path.cwd())
        self.model = None
        self.graph = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._initialized = False

    def _initialize_model(self):
        """初始化模型"""
        if self._initialized:
            return

        try:
            # 根據 config 設定 (dataset, model_type 等) 初始化 Graph 以取得 A (adjacency matrix) 和 parts
            dataset = self.config.get('dataset', 'coco')
            self.graph = Graph(dataset=dataset)

            # 使用 resgcn_model.create 建立模型實例
            model_config = self.config_manager.get_resgcn_config()
            self.model = resgcn_model.create(
                model_type=model_config.get('model_type', 'resgcn'),
                in_channels=model_config.get('in_channels', 3),
                num_class=model_config.get('num_class', 5),
                graph=self.graph
            )

            # 載入預訓練權重 (.pth.tar)，注意處理 module. 前綴 (DataParallel 產生的)
            model_path = self.config.get('model_path')
            if model_path and os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

                # 處理 DataParallel 前綴
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v

                self.model.load_state_dict(new_state_dict)
                logger.info(f"載入模型權重: {model_path}")
            else:
                logger.warning(f"模型權重檔案不存在: {model_path}")

            # 將模型設定為 eval() 模式並移至 GPU (如果可用)
            self.model.eval()
            self.model.to(self.device)

            self._initialized = True
            logger.info("ResGCN模型初始化完成")

        except Exception as e:
            logger.error(f"初始化ResGCN模型失敗: {e}")
            raise

    def predict(self, skeleton_data: np.ndarray) -> Dict[str, Any]:
        """
        預測動作類別

        Args:
            skeleton_data: numpy array，骨架數據

        Returns:
            包含 label, confidence, probs 的字典
        """
        try:
            self._initialize_model()

            # 將資料轉換為 Tensor 並調整維度為 (N, C, T, V, M) (ResGCN 格式)
            # 假設輸入是 (C, T, V) 或 (T, V, C)，需要調整為 (1, C, T, V, 1)
            if skeleton_data.shape[-1] == 3:  # (T, V, C)
                skeleton_data = skeleton_data.transpose(2, 0, 1)  # (C, T, V)

            C, T, V = skeleton_data.shape
            # 補充到標準長度，假設 T=300, V=17
            if T < 300:
                pad_T = 300 - T
                skeleton_data = np.pad(skeleton_data, ((0, 0), (0, pad_T), (0, 0)), mode='constant')
            elif T > 300:
                skeleton_data = skeleton_data[:, :300, :]

            if V < 17:
                pad_V = 17 - V
                skeleton_data = np.pad(skeleton_data, ((0, 0), (0, 0), (0, pad_V)), mode='constant')
            elif V > 17:
                skeleton_data = skeleton_data[:, :, :17]

            # 添加 batch 和 person 維度: (1, C, T, V, 1)
            data = torch.tensor(skeleton_data, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)

            # 直接呼叫 self.model(data) 進行推論
            with torch.no_grad():
                output = self.model(data)
                probs = F.softmax(output, dim=1).cpu().numpy().squeeze()

            # 獲取預測結果
            label = int(np.argmax(probs))
            confidence = float(probs[label])

            return {
                "label": label,
                "confidence": confidence,
                "probs": probs
            }

        except Exception as e:
            logger.error(f"ResGCN預測失敗: {e}")
            raise

    def generate_train_data(self):
        """生成訓練數據 - 使用配置適配器模式"""
        try:
            logger.info("開始生成ResGCN訓練數據...")

            # 獲取適配後的配置
            adapted_config = self.config_manager.get_resgcn_config('generate')

            # 創建臨時config文件
            temp_config_path = self.config_manager.create_temp_config(adapted_config)

            try:
                # 使用臨時config執行
                config_name = Path(temp_config_path).stem

                # 構建命令行參數
                resgcn_main_path = Path("vendor/ResGCNv1/main.py").resolve()
                cmd = [
                    sys.executable,
                    str(resgcn_main_path),
                    "--config", config_name,
                    "--generate_data"
                ]

                # 添加GPU參數
                gpus = self.config.get("gpus", [])
                if gpus:
                    cmd.extend(["--gpus"] + [str(gpu) for gpu in gpus])

                # 設置工作目錄為vendor/ResGCNv1
                resgcn_dir = Path("vendor/ResGCNv1").resolve()

                logger.info(f"執行命令: {' '.join(cmd)}")
                logger.info(f"工作目錄: {resgcn_dir}")

                # 執行命令並實時顯示輸出
                process = subprocess.Popen(
                    cmd,
                    cwd=resgcn_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1
                )

                # 實時讀取並顯示輸出
                output_lines = []
                while True:
                    output_bytes = process.stdout.readline()
                    if output_bytes == b'' and process.poll() is not None:
                        break
                    if output_bytes:
                        try:
                            output = output_bytes.decode('utf-8', errors='replace')
                        except UnicodeDecodeError:
                            output = output_bytes.decode('cp950', errors='replace')  # Windows 中文編碼
                        output = output.strip()
                        output_lines.append(output)
                        logger.info(output)  # 同時記錄到日誌

                returncode = process.poll()
                if returncode != 0:
                    error_msg = '\n'.join(output_lines)
                    logger.error(f"ResGCN數據生成失敗: {error_msg}")
                    raise RuntimeError(f"ResGCN數據生成失敗: {error_msg}")

                logger.info("ResGCN數據生成完成")
                return True

            finally:
                # 清理臨時文件
                Path(temp_config_path).unlink()

        except Exception as e:
            logger.error(f"生成ResGCN數據失敗: {e}")
            raise

    def train(self):
        """訓練模型 - 使用配置適配器模式"""
        try:
            logger.info("開始訓練ResGCN模型...")

            # 獲取適配後的配置
            adapted_config = self.config_manager.get_resgcn_config('train')

            # 創建臨時config文件
            temp_config_path = self.config_manager.create_temp_config(adapted_config)

            try:
                # 使用臨時config執行
                config_name = Path(temp_config_path).stem

                # 構建命令行參數
                resgcn_main_path = Path("vendor/ResGCNv1/main.py").resolve()
                cmd = [
                    sys.executable,
                    str(resgcn_main_path),
                    "--config", config_name
                ]

                # 添加GPU參數
                gpus = self.config.get("gpus", [])
                if gpus:
                    cmd.extend(["--gpus"] + [str(gpu) for gpu in gpus])

                # 設置工作目錄為vendor/ResGCNv1
                resgcn_dir = Path("vendor/ResGCNv1").resolve()

                logger.info(f"執行命令: {' '.join(cmd)}")
                logger.info(f"工作目錄: {resgcn_dir}")

                # 執行命令並實時顯示輸出
                process = subprocess.Popen(
                    cmd,
                    cwd=resgcn_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1
                )

                # 實時讀取並顯示輸出
                output_lines = []
                while True:
                    output_bytes = process.stdout.readline()
                    if output_bytes == b'' and process.poll() is not None:
                        break
                    if output_bytes:
                        try:
                            output = output_bytes.decode('utf-8', errors='replace')
                        except UnicodeDecodeError:
                            output = output_bytes.decode('cp950', errors='replace')  # Windows 中文編碼
                        output = output.strip()
                        output_lines.append(output)
                        logger.info(output)  # 同時記錄到日誌

                returncode = process.poll()
                if returncode != 0:
                    error_msg = '\n'.join(output_lines)
                    logger.error(f"ResGCN訓練失敗: {error_msg}")
                    raise RuntimeError(f"ResGCN訓練失敗: {error_msg}")

                logger.info("ResGCN訓練完成")
                return True

            finally:
                # 清理臨時文件
                Path(temp_config_path).unlink()

        except Exception as e:
            logger.error(f"訓練ResGCN模型失敗: {e}")
            raise

    def _extract_skeleton_data_from_json(self, json_files, label_key: str = "label"):
        """從 SOA JSON 檔案提取骨架數據"""
        # 這裡需要實現從 JSON 檔案讀取骨架數據並準備訓練數據的邏輯
        # 由於 ResGCN 的具體實現可能很複雜，這裡提供一個基本的框架

        skeleton_data_list = []
        labels = []

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 從 frames 欄位提取骨架數據 (SOA 格式: [T, V, C])
                frames = data.get("frames")
                if frames is None:
                    logger.warning(f"JSON 檔案 {json_file} 沒有 frames 資料，跳過")
                    continue

                # 轉換為 numpy 陣列
                skeleton_array = np.array(frames)
                skeleton_data_list.append(skeleton_array)

                # 直接從 metadata 中根據 label_key 取得標籤
                metadata = data.get("metadata", {})
                label = metadata.get(label_key, 0)
                labels.append(label)

            except Exception as e:
                logger.warning(f"處理 JSON 檔案 {json_file} 時發生錯誤: {e}")
                continue

        # 儲存處理後的數據
        self.skeleton_data = skeleton_data_list
        self.labels = labels

        logger.info(f"從 {len(json_files)} 個 JSON 檔案提取到 {len(skeleton_data_list)} 筆有效骨架數據")
