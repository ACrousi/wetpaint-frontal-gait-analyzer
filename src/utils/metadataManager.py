import pandas as pd
import os
import logging
from pathlib import Path

# from config import ConfigManager


class MetadataManager:
    """元數據管理器 - 重構後版本
    
    專注於核心元數據管理功能：
    1. 載入和儲存 CSV 元數據
    2. 記錄的 CRUD 操作
    3. 資料過濾和篩選
    """
    
    def __init__(self, config: dict):
        """
        初始化 MetadataManager

        Args:
            config: 配置字典
            filter_conditions: 用於篩選 DataFrame 的條件
        """
        self.metadata_csv = Path(config.get("metadata_path", None))
        self.label_column = config.get("label_column", "age_months_range")  # 從 config 讀取 label_column，預設為 "age_months_range"
        logging.info(f"元數據 CSV 路徑: {self.metadata_csv}")
        logging.info(f"標籤欄位: {self.label_column}")
        # self.deuplicate = config.get("deuplicate", False)

        # 初始化空的 DataFrame
        self.df = pd.DataFrame()

        # 載入現有元數據
        if self.metadata_csv and os.path.isfile(self.metadata_csv):
            try:
                self.df = pd.read_csv(self.metadata_csv)
                self.filter_df = self.df.copy(deep=True)
                logging.info(f"已載入現有元數據: {self.metadata_csv}")
                
                # 應用篩選條件
                if config.get("filter_conditions", False):
                    self._apply_filter_conditions(config.get("filter_conditions"))

            except Exception as e:
                logging.error(f"載入元數據時發生錯誤 {self.metadata_csv}: {e}")
                self.df = pd.DataFrame()

    def _apply_filter_conditions(self, filter_conditions):
        """應用篩選條件"""
        logging.info(f"應用篩選條件: {filter_conditions}")
        try:
            if isinstance(filter_conditions, dict):
                for column, condition in filter_conditions.items():
                    if isinstance(condition, (list, tuple)):
                        self.filter_df = self.filter_df[self.filter_df[column].isin(condition)]
                    else:
                        self.filter_df = self.filter_df[self.filter_df[column] == condition]
            elif isinstance(filter_conditions, list):
                for condition in filter_conditions:
                    self.filter_df = self.filter_df.query(condition)
            
            self.df = self.filter_df.copy(deep=True)
            logging.info(f"成功應用篩選條件到元數據")
            
        except Exception as e:
            logging.error(f"應用篩選條件時發生錯誤: {e}")
            logging.info(f"由於篩選錯誤，回復到原始元數據")

    def has_record(self, video_file):
        """檢查記錄中是否已存在該影片（支援完整路徑或檔名）"""
        if self.df.empty:
            return False
        # 支援完整路徑比對或檔名比對
        video_file_name = Path(video_file).name
        return (
            (self.df["original_video"] == video_file) | 
            (self.df["original_video"].apply(lambda x: Path(x).name if pd.notna(x) else '') == video_file_name)
        ).any()

    def get_record(self, video_file):
        """根據影片檔名或路徑取得所有符合的記錄"""
        if self.df.empty or not self.has_record(video_file):
            return {}

        video_file_name = Path(video_file).name
        matching_records = self.df.loc[
            (self.df["original_video"] == video_file) | 
            (self.df["original_video"].apply(lambda x: Path(x).name if pd.notna(x) else '') == video_file_name)
        ]
        if len(matching_records) == 1:
            return matching_records.iloc[0].to_dict()
        else:
            return matching_records.to_dict('records')

    def get_record_by_index(self, idx: int):
        """根據索引取得記錄"""
        if self.df.empty or idx >= len(self.df):
            return {}
        return self.df.iloc[idx].to_dict()

    def add_record(self, video_path, metadata_dict):
        """新增一筆記錄"""
        new_record = {"original_video": video_path, **metadata_dict}
        new_row = pd.DataFrame([new_record])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        logging.info(f"已新增記錄: {video_path}")

    def update_record(self, idx: int, updates: dict):
        """更新指定索引的記錄"""
        if idx >= len(self.df):
            logging.warning(f"索引 {idx} 超出範圍")
            return False
            
        for key, value in updates.items():
            self.df.at[idx, key] = value
            
        logging.info(f"已更新記錄 {idx}: {list(updates.keys())}")
        return True

    def update_csv_paths(self, idx, csv_paths):
        """更新 CSV 檔案路徑"""
        if not csv_paths:
            return
            
        if len(csv_paths) == 1:
            self.df.at[idx, 'output_csv'] = csv_paths[0]
            self.df.at[idx, 'keypoint_file'] = os.path.basename(csv_paths[0])
        else:
            # 保存第一個路徑到當前記錄
            self.df.at[idx, 'output_csv'] = csv_paths[0]
            self.df.at[idx, 'keypoint_file'] = os.path.basename(csv_paths[0])

            # 為每個額外的 CSV 路徑創建新記錄
            current_row = self.df.iloc[idx].to_dict()
            
            for csv_path in csv_paths[1:]:
                new_record = current_row.copy()
                new_record['output_csv'] = csv_path
                new_record['keypoint_file'] = os.path.basename(csv_path)

                self.df = pd.concat([
                    self.df,
                    pd.DataFrame([new_record])
                ], ignore_index=True)

    def update_total_frame(self, idx: int, total_frame: int):
        """更新總幀數"""
        if idx < len(self.df):
            self.df.at[idx, 'totalframe'] = total_frame
            logging.info(f"已更新索引 {idx} 的總幀數: {total_frame}")
        else:
            logging.warning(f"索引 {idx} 超出範圍，無法更新總幀數")

    def merge_labels(self, label_csv_path, key_field, fields_to_merge):
        """從標籤 CSV 檔案合併資料到現有記錄"""
        if not os.path.exists(label_csv_path):
            logging.info(f"標籤檔案不存在: {label_csv_path}")
            return False

        try:
            labels_df = pd.read_csv(label_csv_path)
            cols_to_use = [key_field] + [f for f in fields_to_merge if f in labels_df.columns]
            labels_df = labels_df[cols_to_use]

            if key_field in self.df.columns:
                self.df = pd.merge(
                    self.df,
                    labels_df,
                    on=key_field,
                    how='left'
                )
                logging.info(f"已從 {label_csv_path} 合併 {len(labels_df)} 筆記錄")
                return True
            else:
                logging.info(f"元數據中找不到關鍵欄位 '{key_field}'")
                return False
                
        except Exception as e:
            logging.error(f"從 {label_csv_path} 合併標籤時發生錯誤: {e}")
            return False

    def save(self, output_directory, filename="metadata.csv"):
        """儲存 DataFrame 到 CSV 檔案"""
        if self.metadata_csv:
            metadata_filename = os.path.splitext(os.path.basename(self.metadata_csv))[0]
            base, ext = os.path.splitext(filename)
            new_filename = f"{metadata_filename}_{base}{ext}"
        else:
            new_filename = filename
            
        output_path = os.path.join(output_directory, new_filename)

        # 確保輸出目錄存在
        Path(output_directory).mkdir(parents=True, exist_ok=True)

        # 儲存處理後的資料
        self.df_out = self.df.copy(deep=True)
        self.df_out = self.df_out.sort_values(by='original_video')
        self.df_out.to_csv(output_path, encoding='utf-8-sig', index=False)
        logging.info(f"元數據已儲存至: {output_path}")

    def filter_records(self, conditions: dict):
        """根據條件篩選記錄"""
        filtered_df = self.df.copy()
        
        for column, condition in conditions.items():
            if column not in filtered_df.columns:
                logging.warning(f"欄位 '{column}' 不存在於元數據中")
                continue
                
            if isinstance(condition, (list, tuple)):
                filtered_df = filtered_df[filtered_df[column].isin(condition)]
            else:
                filtered_df = filtered_df[filtered_df[column] == condition]
                
        return filtered_df

    def get_video_info(self, idx: int):
        """取得影片資訊（簡化版本，移除 COCO 相關）"""
        try:
            if idx >= len(self.df):
                return None

            row = self.df.iloc[idx]
            video_file_path = Path(row['original_video'])

            if not video_file_path.exists():
                logging.warning(f"影片檔案不存在：{video_file_path}")
                return None

            video_info = {
                'index': idx,
                'case_id': row.get('id', ''),
                'video_path': row.get('original_video', ''),
                'video_name': video_file_path.name,  # 從 original_video 路徑提取檔名
                'label': row.get(self.label_column, None),  # 添加 label 欄位
                'age_months': row.get('age_months', None),
                'development_result': row.get('development_result', None),
            }

            # 添加其他可用的欄位
            for col in self.df.columns:
                if col not in video_info:
                    video_info[col] = row.get(col, None)

            return video_info
        except Exception as e:
            logging.error(f"提取影片資訊時發生錯誤（索引 {idx}）：{e}")
            return None

    def get_all_video_info(self):
        """取得所有影片的資訊列表"""
        video_list = []
        for idx in range(len(self.df)):
            video_info = self.get_video_info(idx)
            if video_info is not None:
                video_list.append(video_info)
        logging.info(f"已取得 {len(video_list)} 筆影片資訊")
        return video_list

    def get_statistics(self):
        """取得元數據統計資訊"""
        if self.df.empty:
            return {
                "total_records": 0,
                "unique_videos": 0,
                "columns": []
            }
            
        stats = {
            "total_records": len(self.df),
            "unique_videos": self.df['original_video'].nunique() if 'original_video' in self.df.columns else 0,
            "columns": list(self.df.columns),
            "data_types": self.df.dtypes.to_dict()
        }
        
        # 添加每個欄位的統計資訊
        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64']:
                stats[f"{col}_stats"] = {
                    "mean": self.df[col].mean(),
                    "min": self.df[col].min(),
                    "max": self.df[col].max(),
                    "null_count": self.df[col].isnull().sum()
                }
            else:
                stats[f"{col}_stats"] = {
                    "unique_count": self.df[col].nunique(),
                    "null_count": self.df[col].isnull().sum(),
                    "most_common": self.df[col].mode().iloc[0] if not self.df[col].mode().empty else None
                }
                
        return stats

    def remove_records(self, indices: list):
        """移除指定索引的記錄"""
        if not indices:
            return
            
        # 確保索引在有效範圍內
        valid_indices = [idx for idx in indices if 0 <= idx < len(self.df)]
        if len(valid_indices) != len(indices):
            logging.warning(f"某些索引超出範圍，將移除有效的 {len(valid_indices)} 筆記錄")
            
        self.df = self.df.drop(self.df.index[valid_indices]).reset_index(drop=True)
        logging.info(f"已移除 {len(valid_indices)} 筆記錄")

    def duplicate_check(self):
        """檢查重複記錄"""
        if 'original_video' in self.df.columns:
            duplicates = self.df[self.df.duplicated(subset=['original_video'], keep=False)]
            if not duplicates.empty:
                logging.warning(f"發現 {len(duplicates)} 筆重複記錄")
                return duplicates
        return pd.DataFrame()

    def backup_metadata(self, backup_path: str):
        """備份當前元數據"""
        backup_path = Path(backup_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(backup_path, encoding='utf-8-sig', index=False)
        logging.info(f"元數據已備份至: {backup_path}")

    def load_backup(self, backup_path: str):
        """從備份載入元數據"""
        if not os.path.exists(backup_path):
            logging.error(f"備份檔案不存在: {backup_path}")
            return False
            
        try:
            self.df = pd.read_csv(backup_path)
            logging.info(f"已從備份載入元數據: {backup_path}")
            return True
        except Exception as e:
            logging.error(f"載入備份時發生錯誤: {e}")
            return False