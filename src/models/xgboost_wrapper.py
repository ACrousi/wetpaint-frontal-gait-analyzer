import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json

logger = logging.getLogger(__name__)


class XGBoostWrapper:
    """XGBoost模型封裝器，用於步態分析預測"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化XGBoostWrapper

        Args:
            config: XGBoost配置字典
        """
        self.config = config
        self.model = None
        self.metadata_df = None
        self._initialized = False

    def generate_train_data(self):
        """生成訓練數據 - 從 metadata CSV 的 exported_paths 欄位讀取 JSON 檔案"""
        try:
            logger.info("開始生成XGBoost訓練數據...")

            # 從 metadata_path 讀取 CSV
            metadata_path = self.config.get("metadata_path")
            if not metadata_path:
                raise ValueError("未設置 metadata_path")

            metadata_df = pd.read_csv(metadata_path)
            logger.info(f"從 {metadata_path} 載入 metadata，形狀: {metadata_df.shape}")

            # 從 exported_paths 欄位獲取 JSON 文件列表
            if 'exported_paths' not in metadata_df.columns:
                raise ValueError("metadata CSV 中沒有 'exported_paths' 欄位")

            json_files = []
            for path in metadata_df['exported_paths'].dropna():
                json_path = Path(path)
                if json_path.exists():
                    json_files.append(json_path)
                else:
                    logger.warning(f"JSON 檔案不存在: {json_path}")

            if not json_files:
                raise ValueError("沒有找到有效的 JSON 檔案")

            logger.info(f"從 metadata 找到 {len(json_files)} 個有效的 JSON 檔案")

            # 從 JSON 檔案提取特徵和標籤
            self._extract_features_from_json(json_files)

            logger.info("XGBoost數據生成完成")
            return True

        except Exception as e:
            logger.error(f"生成XGBoost數據失敗: {e}")
            raise

    def train(self, n_splits=5):
        """
        訓練XGBoost模型

        Args:
            n_splits: K-Fold 分割數量，當為1時使用簡單分割
        """
        try:
            logger.info("開始訓練XGBoost模型...")

            if self.metadata_df is None:
                raise ValueError("請先調用generate_data()生成訓練數據")

            # 準備訓練數據
            X, y, feature_names = self._prepare_training_data()

            # 使用K-Fold交叉驗證訓練
            results = self._train_with_kfold(X, y, feature_names, n_splits)

            # 保存訓練結果
            self._save_training_results(results)

            logger.info("XGBoost模型訓練完成")
            return results

        except Exception as e:
            logger.error(f"訓練XGBoost模型失敗: {e}")
            raise

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        預測步態類別

        Args:
            features: 外部轉換好的特徵陣列 (n_features,)

        Returns:
            預測結果，格式為 {"probs": np.ndarray, "label": int}
        """
        try:
            if self.model is None:
                self._load_model()

            # 確保特徵格式正確
            if features.ndim == 1:
                features = features.reshape(1, -1)
            elif features.ndim > 2:
                raise ValueError("特徵陣列維度不正確")

            # 預測
            probs = self.model.predict_proba(features)[0]
            label = int(np.argmax(probs))

            return {
                "probs": probs,
                "label": label
            }

        except Exception as e:
            logger.error(f"XGBoost預測失敗: {e}")
            raise

    def _extract_features_from_json(self, json_files, label_key: str = "label"):
        """從 SOA JSON 檔案提取特徵"""
        features_list = []
        labels = []

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 從 features 欄位提取特徵
                features_dict = data.get("features", {})
                if not features_dict:
                    logger.warning(f"JSON 檔案 {json_file} 沒有 features 資料，跳過")
                    continue

                # 從 config 中獲取特徵清單
                features_config = self.config.get("features", list(features_dict.keys()))
                features = [float(features_dict.get(feature, 0)) for feature in features_config]
                features_list.append(features)

                # 直接從 metadata 中根據 label_key 取得標籤
                metadata = data.get("metadata", {})
                label = metadata.get(label_key, 0)
                labels.append(label)

            except Exception as e:
                logger.warning(f"處理 JSON 檔案 {json_file} 時發生錯誤: {e}")
                continue

        # 創建 DataFrame 方便後續處理
        self.metadata_df = pd.DataFrame(features_list, columns=self.config.get("features", []))
        self.metadata_df['label'] = labels

        logger.info(f"從 {len(json_files)} 個 JSON 檔案提取到 {len(features_list)} 筆有效資料")

    def _prepare_training_data(self):
        """準備訓練數據"""
        # 從 metadata_df 中提取特徵和標籤
        feature_columns = [col for col in self.metadata_df.columns if col != 'label']
        X = self.metadata_df[feature_columns].values
        y = self.metadata_df['label'].values

        logger.info(f"訓練數據形狀: X={X.shape}, y={y.shape}")
        return X, y, feature_columns

    def _train_with_kfold(self, X, y, feature_names, n_splits=5):
        """
        使用K-Fold交叉驗證訓練模型

        Args:
            X: 特徵矩陣
            y: 標籤向量
            feature_names: 特徵名稱列表
            n_splits: 分割數量

        Returns:
            訓練結果字典
        """
        if n_splits == 1:
            # 當 n_splits=1 時，使用簡單分割
            logger.info("使用簡單train-test分割...")
            return self._train_simple_split(X, y, feature_names)

        # 使用StratifiedKFold確保每個fold中類別分佈相同
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_results = []
        best_model = None
        best_score = 0

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            logger.info(f"訓練 Fold {fold_idx}/{n_splits}...")

            # 分割數據
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 創建並訓練模型
            model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=len(np.unique(y)),
                max_depth=self.config.get("max_depth", 6),
                learning_rate=self.config.get("learning_rate", 0.1),
                n_estimators=self.config.get("n_estimators", 100),
                early_stopping_rounds=self.config.get("early_stopping_rounds", 10),
                random_state=42
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # 設置特徵名稱
            model.get_booster().feature_names = feature_names

            # 評估模型
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)

            # 計算多個評估指標
            metrics = {
                'fold': fold_idx,
                'accuracy': accuracy_score(y_val, y_pred),
                'f1_macro': f1_score(y_val, y_pred, average='macro'),
                'precision_macro': precision_score(y_val, y_pred, average='macro'),
                'recall_macro': recall_score(y_val, y_pred, average='macro')
            }

            fold_results.append(metrics)
            logger.info(f"Fold {fold_idx} - Accuracy: {metrics['accuracy']:.4f}, "
                       f"F1: {metrics['f1_macro']:.4f}")

            # 保存最佳模型
            if metrics['accuracy'] > best_score:
                best_score = metrics['accuracy']
                best_model = model
                best_fold = fold_idx

        # 計算平均指標
        avg_metrics = {
            'accuracy': np.mean([r['accuracy'] for r in fold_results]),
            'accuracy_std': np.std([r['accuracy'] for r in fold_results]),
            'f1_macro': np.mean([r['f1_macro'] for r in fold_results]),
            'f1_std': np.std([r['f1_macro'] for r in fold_results]),
            'precision_macro': np.mean([r['precision_macro'] for r in fold_results]),
            'recall_macro': np.mean([r['recall_macro'] for r in fold_results])
        }

        logger.info(f"\n{'='*50}")
        logger.info(f"K-Fold 交叉驗證結果摘要:")
        logger.info(f"平均準確率: {avg_metrics['accuracy']:.4f} ± {avg_metrics['accuracy_std']:.4f}")
        logger.info(f"平均 F1 分數: {avg_metrics['f1_macro']:.4f} ± {avg_metrics['f1_std']:.4f}")
        logger.info(f"最佳模型來自 Fold {best_fold}")
        logger.info(f"{'='*50}\n")

        # 保存最佳模型
        self.model = best_model
        model_path = Path(self.config.get("model_path", "./models/xgboost_model.json"))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(model_path))
        logger.info(f"最佳模型已保存至: {model_path}")

        # 生成圖表（使用完整訓練集）
        self._generate_plots(X, feature_names)

        return {
            'fold_results': fold_results,
            'avg_metrics': avg_metrics,
            'best_fold': best_fold,
            'best_score': best_score
        }

    def _train_simple_split(self, X, y, feature_names):
        """
        使用簡單train-test分割訓練模型（當 n_splits=1 時使用）
        """
        X_train, X_val, y_train, y_val = StratifiedKFold(
            n_splits=2, shuffle=True, random_state=42
        ).split(X, y).__next__()

        # 創建XGBoost模型
        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=len(np.unique(y)),
            max_depth=self.config.get("max_depth", 6),
            learning_rate=self.config.get("learning_rate", 0.1),
            n_estimators=self.config.get("n_estimators", 100),
            early_stopping_rounds=self.config.get("early_stopping_rounds", 10),
            random_state=42
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=True
        )

        self.model.get_booster().feature_names = feature_names

        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        logger.info(f"驗證準確率: {accuracy:.4f}")

        # 保存模型
        model_path = Path(self.config.get("model_path", "./models/xgboost_model.json"))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(model_path))

        self._generate_plots(X_train, feature_names)

        return {
            'accuracy': accuracy,
            'method': 'simple_split'
        }

    def _save_training_results(self, results):
        """保存訓練結果到JSON文件"""
        results_path = Path(self.config.get("results_path", "./outputs/training_results.json"))
        results_path.parent.mkdir(parents=True, exist_ok=True)

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"訓練結果已保存至: {results_path}")

    def _extract_single_features(self, gait_data: Dict[str, Any]) -> list:
        """從單個步態數據提取特徵"""
        # 實現單個樣本的特徵提取邏輯
        return list(gait_data.values())

    def _load_model(self):
        """載入已訓練的模型"""
        model_path = Path(self.config.get("model_path", "./models/xgboost_model.json"))
        if model_path.exists():
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(model_path))
            logger.info(f"載入模型: {model_path}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

    def _generate_plots(self, X_data: np.ndarray, feature_names: list):
        """
        生成特徵重要性和SHAP圖 (長條圖 + 每個類別的蜂群圖)
        
        Args:
            X_data: 用於計算 SHAP 值的特徵矩陣 (例如 X_train 或
                    K-Fold 結束後的完整 X)
            feature_names: 特徵名稱列表
        """
        try:
            import matplotlib.pyplot as plt
            # 關閉matplotlib的debug日誌
            import logging
            logging.getLogger('matplotlib').setLevel(logging.WARNING)
            logging.getLogger('PIL').setLevel(logging.WARNING)

            # --- 1. 繪製特徵重要性 ---
            fig, ax = plt.subplots(figsize=(12, 8)) # 增大圖表尺寸
            xgb.plot_importance(self.model, max_num_features=20, ax=ax)
            ax.set_title('Feature Importance (Top 20)')
            plt.tight_layout() # 自動調整佈局
            feature_importance_path = self.config.get("feature_importance_output_path", "./outputs/feature_importance.png")
            Path(feature_importance_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(feature_importance_path, bbox_inches='tight') # 確保完整保存
            plt.close()
            logger.info(f"特徵重要性圖已保存至: {feature_importance_path}")

            # --- 2. 繪製SHAP圖 ---
            import shap
            logger.info("正在計算 SHAP values (這可能需要一些時間)...")
            
            # 創建 Explainer
            # 使用 shap.maskers.Independent 來避免舊版 shap 的警告
            masker = shap.maskers.Independent(X_data, max_samples=100)
            explainer = shap.Explainer(self.model.predict_proba, masker)
            shap_values = explainer(X_data)
            
            logger.info("SHAP values 計算完成。")

            # --- 2a. 繪製多類別總結長條圖 (你看到的原始圖) ---
            logger.info("正在生成多類別 SHAP 總結長條圖...")
            shap.summary_plot(shap_values, X_data, feature_names=feature_names, show=False, plot_type='bar')
            
            base_shap_path_str = self.config.get("shap_output_path", "./outputs/shap_summary.png")
            base_shap_path = Path(base_shap_path_str)
            
            # 確保檔案名稱是總結圖的名稱
            shap_bar_path = base_shap_path.parent / f"{base_shap_path.stem}_bar_summary.png"
            if base_shap_path.name == "shap_summary.png": # 處理預設情況
                 shap_bar_path = base_shap_path
            
            plt.title("SHAP Multi-Class Summary (Overall Importance)")
            plt.savefig(shap_bar_path, bbox_inches='tight')
            plt.close()
            logger.info(f"SHAP多類別總結長條圖已保存至: {shap_bar_path}")

            # --- 2b. 繪製每個類別的蜂群圖 (Beeswarm plots) ---
            
            # 獲取類別標籤，`self.model.classes_` 儲存了訓練時的原始標籤 (例如 0, 1, 2)
            class_labels = self.model.classes_
            
            logger.info(f"正在為 {len(class_labels)} 個類別生成 SHAP 蜂群圖...")

            for i, class_label in enumerate(class_labels):
                logger.info(f"  正在生成 Class: {class_label} (Index: {i}) 的圖...")
                
                # 從 3D SHAP Explanation 物件中選取特定類別的 SHAP values
                # shap_values[..., i] 會返回一個 (n_samples, n_features) 的 Explanation 物件
                shap.summary_plot(
                    shap_values[..., i], 
                    X_data, 
                    feature_names=feature_names, 
                    show=False,
                    plot_type="dot" # 明確指定 'dot' (beeswarm)
                )
                
                # 建立新的檔案路徑
                shap_class_path = base_shap_path.parent / f"{base_shap_path.stem}_class_{class_label}_beeswarm.png"
                
                plt.title(f"SHAP Summary (Beeswarm) for Class: {class_label}")
                plt.savefig(shap_class_path, bbox_inches='tight')
                plt.close()
                logger.info(f"  SHAP Class {class_label} 蜂群圖已保存至: {shap_class_path}")

        except ImportError as ie:
             logger.error(f"生成圖表失敗：缺少必要的套件。請安裝 'matplotlib' 和 'shap'。錯誤: {ie}")
             # 不 re-raise，這樣即使沒有繪圖套件，訓練也可以繼續
        except Exception as e:
            logger.error(f"生成圖表時發生未預期的錯誤: {e}")
            # 也不 re-raise