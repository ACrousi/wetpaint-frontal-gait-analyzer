import os
import pickle
import logging
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import utils as U
from .preprocessor import pre_normalization

class COCO_Generator():
    def __init__(self, args, dataset_args):
        self.num_person_out = 1  # Single person
        self.num_joint = 17      # COCO has 17 joints
        self.max_frame = 150     # Maximum frames
        self.dataset = 'coco'
        self.print_bar = not args.no_progress_bar
        self.generate_label = args.generate_label

        # 新增：模型類型判斷
        self.model_type = args.model_type  # 'gaitmlp', 'resgcn', 'fusion'

        # Gait/Features 參數設定
        self.use_gait = dataset_args.get('use_gait', False)
        self.gait_columns = dataset_args.get('gait_columns', [])
        self.input_path = dataset_args['input_path']

        # Seed for shuffling case_ids
        self.seed = dataset_args.get('seed', 0)

        # 輸出路徑
        self.out_path = '{}/coco'.format(dataset_args['path'])
        U.create_folder(self.out_path)

        # 讀取 metadata CSV
        metadata_path = dataset_args.get('metadata_path', './outputs/2025.csv')
        self.metadata_df = pd.read_csv(metadata_path)
        logging.info(f"Loaded metadata from {metadata_path}, shape: {self.metadata_df.shape}")

        # 獲取檔案列表
        self.file_list = self.metadata_df['keypoint_filename'].unique().tolist()
        logging.info(f"Found {len(self.file_list)} unique keypoint files")

    def start(self):
        """生成三組資料：retard, train, eval"""
        # 分離 retard 和 normal 資料
        if 'development_result' in self.metadata_df.columns:
            retard_df = self.metadata_df[self.metadata_df['development_result'] != 0]
            normal_df = self.metadata_df[self.metadata_df['development_result'] == 0]
            logging.info(f"資料分離：retard={len(retard_df)}, normal={len(normal_df)}")
        else:
            logging.warning("'development_result' 欄位不存在，將所有資料視為 normal")
            retard_df = pd.DataFrame()
            normal_df = self.metadata_df

        # 生成 retard 資料
        if len(retard_df) > 0:
            logging.info(f'Phase: retard ({len(retard_df)} samples)')
            self.gendata_for_df(retard_df, 'retard')
        else:
            logging.info("無 retard 資料，跳過生成")

        # 生成 train/eval 資料（從 normal_df）
        if len(normal_df) > 0:
            self.gendata_train_eval(normal_df)
        else:
            logging.warning("無 normal 資料，無法生成 train/eval")

    def gendata_for_df(self, df, phase):
        """從指定 DataFrame 生成資料"""
        file_list = df['keypoint_filename'].unique().tolist()
        
        sample_files = []
        sample_labels = []

        # Extract labels from metadata (age_months for regression)
        for file_name in file_list:
            metadata_row = df[df['keypoint_filename'] == file_name]
            if not metadata_row.empty:
                age_months = metadata_row['age_months'].iloc[0] if 'age_months' in metadata_row.columns else metadata_row['age_months_range'].iloc[0]
                label = float(age_months) if pd.notna(age_months) else 0.0
                sample_files.append(file_name)
                sample_labels.append(label)
            else:
                logging.warning(f"No metadata found for {file_name}, skipping")

        # 根據模型類型生成資料
        self._generate_by_model_type(phase, sample_files, sample_labels, df)

    def gendata_train_eval(self, normal_df):
        """從 normal 資料生成 train/eval（按 case_id 分割）"""
        file_list = normal_df['keypoint_filename'].unique().tolist()
        
        # Group files by case_id
        case_groups = {}
        for file_name in file_list:
            metadata_row = normal_df[normal_df['keypoint_filename'] == file_name]
            if not metadata_row.empty:
                case_id = metadata_row['case_id'].iloc[0]
                if case_id not in case_groups:
                    case_groups[case_id] = []
                case_groups[case_id].append(file_name)

        # Split cases: 80% for training, 20% for evaluation
        case_ids = list(case_groups.keys())

        # Shuffle case_ids
        if self.seed is not None:
            np.random.seed(self.seed)
            np.random.shuffle(case_ids)

        num_cases = len(case_ids)
        train_split = int(num_cases * 0.8)

        train_case_ids = case_ids[:train_split]
        eval_case_ids = case_ids[train_split:]

        logging.info(f"Case split: train={len(train_case_ids)}, eval={len(eval_case_ids)}")

        # 生成 train 資料
        train_files = []
        for case_id in train_case_ids:
            train_files.extend(case_groups[case_id])
        
        train_labels = self._extract_labels(train_files, normal_df)
        logging.info(f'Phase: train ({len(train_files)} samples)')
        self._generate_by_model_type('train', train_files, train_labels, normal_df)
        self._log_label_statistics('train', train_files, train_labels, train_case_ids, normal_df)

        # 生成 eval 資料
        eval_files = []
        for case_id in eval_case_ids:
            eval_files.extend(case_groups[case_id])
        
        eval_labels = self._extract_labels(eval_files, normal_df)
        logging.info(f'Phase: eval ({len(eval_files)} samples)')
        self._generate_by_model_type('eval', eval_files, eval_labels, normal_df)
        self._log_label_statistics('eval', eval_files, eval_labels, eval_case_ids, normal_df)

    def _extract_labels(self, file_list, df):
        """從檔案列表提取標籤"""
        labels = []
        for file_name in file_list:
            metadata_row = df[df['keypoint_filename'] == file_name]
            if not metadata_row.empty:
                age_months = metadata_row['age_months'].iloc[0] if 'age_months' in metadata_row.columns else metadata_row['age_months_range'].iloc[0]
                label = float(age_months) if pd.notna(age_months) else 0.0
                labels.append(label)
            else:
                labels.append(0.0)
        return labels

    def _generate_by_model_type(self, phase, sample_files, sample_labels, df):
        """根據模型類型生成對應資料"""
        if self.model_type == 'gaitmlp':
            self._generate_gait_only(phase, sample_files, sample_labels)
        elif 'resgcn' in self.model_type and not self.use_gait:
            self._generate_frames_only(phase, sample_files, sample_labels)
        elif 'resgcn' in self.model_type and self.use_gait:
            self._generate_frames_and_gait(phase, sample_files, sample_labels)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def read_xyz(self, keypoint_filename):
        """讀取 SOA 格式的 JSON 骨架資料"""
        # 從 metadata 找到對應的 exported_paths
        metadata_row = self.metadata_df[self.metadata_df['keypoint_filename'] == keypoint_filename]
        if metadata_row.empty:
            raise ValueError(f"No metadata found for keypoint_filename: {keypoint_filename}")

        keypoint_json_path = os.path.join(self.input_path, keypoint_filename)

        if not os.path.exists(keypoint_json_path):
            raise FileNotFoundError(f"Keypoint JSON file not found: {keypoint_json_path}")

        # 讀取 JSON（SOA 格式）
        with open(keypoint_json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # 提取 frames: [T, V, C]
        frames_soa = np.array(json_data['frames'])
        T, V, C = frames_soa.shape

        if V != self.num_joint:
            raise ValueError(f"Expected {self.num_joint} joints, got {V}")

        # 建立資料張量 (C, T, V, M)
        data = np.zeros((3, self.max_frame, self.num_joint, self.num_person_out), dtype=np.float32)

        # 填入 x, y 座標：從 (T, V, 2) 轉為 (2, T, V)
        actual_frames = min(T, self.max_frame)
        data[:2, :actual_frames, :, 0] = frames_soa[:actual_frames, :, :2].transpose(2, 0, 1)

        return data

    def read_features(self, keypoint_filename):
        """讀取 JSON 的 features 欄位作為步態特徵"""
        # 從 metadata 找到對應的 exported_paths
        metadata_row = self.metadata_df[self.metadata_df['keypoint_filename'] == keypoint_filename]
        if metadata_row.empty:
            raise ValueError(f"No metadata found for keypoint_filename: {keypoint_filename}")

        keypoint_json_path = os.path.join(self.input_path, keypoint_filename)

        if not os.path.exists(keypoint_json_path):
            raise FileNotFoundError(f"Keypoint JSON file not found: {keypoint_json_path}")

        # 讀取 JSON
        with open(keypoint_json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # 提取 features
        features = json_data.get('features', {})

        # 根據 gait_columns 提取對應特徵
        gait_values = []
        for col in self.gait_columns:
            val = features.get(col, 0.0)
            if pd.notna(val):
                gait_values.append(float(val))
            else:
                gait_values.append(0.0)

        return np.array(gait_values, dtype=np.float32)

    def _generate_gait_only(self, phase, sample_files, sample_labels):
        """模式 1：僅生成步態特徵"""
        with open(f'{self.out_path}/{phase}_label.pkl', 'wb') as f:
            pickle.dump((sample_files, list(sample_labels)), f)

        if not self.generate_label:
            gait_data = []
            items = tqdm(sample_files, dynamic_ncols=True) if self.print_bar else sample_files
            for file_name in items:
                try:
                    features = self.read_features(file_name)
                    gait_data.append(features)
                except Exception as e:
                    logging.error(f"Error reading features from {file_name}: {e}")
                    gait_data.append(np.zeros(len(self.gait_columns), dtype=np.float32))

            gait_array = np.array(gait_data)

            # 標準化
            if gait_array.shape[0] > 1:
                gait_mean = np.mean(gait_array, axis=0)
                gait_std = np.std(gait_array, axis=0)
                gait_std = np.where(gait_std == 0, 1.0, gait_std)
                gait_array = (gait_array - gait_mean) / gait_std
                logging.info(f"Standardized gait - Mean: {gait_mean}, Std: {gait_std}")

            np.save(f'{self.out_path}/{phase}_gait.npy', gait_array)
            logging.info(f"Saved gait-only data, shape: {gait_array.shape}")

    def _generate_frames_only(self, phase, sample_files, sample_labels):
        """模式 2：僅生成骨架序列"""
        with open(f'{self.out_path}/{phase}_label.pkl', 'wb') as f:
            pickle.dump((sample_files, list(sample_labels)), f)

        if not self.generate_label:
            fp = np.zeros((len(sample_labels), 3, self.max_frame, self.num_joint, self.num_person_out), dtype=np.float32)

            items = tqdm(sample_files, dynamic_ncols=True) if self.print_bar else sample_files
            for i, file_name in enumerate(items):
                try:
                    data = self.read_xyz(file_name)
                    fp[i, :, :, :, :] = data
                except Exception as e:
                    logging.error(f"Error reading {file_name}: {e}")
                    continue

            np.save(f'{self.out_path}/{phase}_data.npy', fp)
            logging.info(f"Saved frames-only data, shape: {fp.shape}")

    def _generate_frames_and_gait(self, phase, sample_files, sample_labels):
        """模式 3：生成骨架序列 + 步態特徵"""
        with open(f'{self.out_path}/{phase}_label.pkl', 'wb') as f:
            pickle.dump((sample_files, list(sample_labels)), f)

        if not self.generate_label:
            # 生成 frames
            fp = np.zeros((len(sample_labels), 3, self.max_frame, self.num_joint, self.num_person_out), dtype=np.float32)
            gait_data = []

            items = tqdm(sample_files, dynamic_ncols=True) if self.print_bar else sample_files
            for i, file_name in enumerate(items):
                try:
                    # 讀取 frames
                    data = self.read_xyz(file_name)
                    fp[i, :, :, :, :] = data

                    # 讀取 features
                    features = self.read_features(file_name)
                    gait_data.append(features)
                except Exception as e:
                    logging.error(f"Error reading {file_name}: {e}")
                    gait_data.append(np.zeros(len(self.gait_columns), dtype=np.float32))
                    continue

            # 儲存 frames
            np.save(f'{self.out_path}/{phase}_data.npy', fp)
            logging.info(f"Saved frames data, shape: {fp.shape}")

            # 標準化並儲存 gait
            gait_array = np.array(gait_data)
            if gait_array.shape[0] > 1:
                gait_mean = np.mean(gait_array, axis=0)
                gait_std = np.std(gait_array, axis=0)
                gait_std = np.where(gait_std == 0, 1.0, gait_std)
                gait_array = (gait_array - gait_mean) / gait_std
                logging.info(f"Standardized gait - Mean: {gait_mean}, Std: {gait_std}")

            np.save(f'{self.out_path}/{phase}_gait.npy', gait_array)
            logging.info(f"Saved gait data, shape: {gait_array.shape}")

    def _log_label_statistics(self, phase, sample_files, sample_labels, selected_case_ids, df):
        """Log statistics for each label: number of case_ids and segments"""
        from collections import defaultdict

        # Create mapping from file to case_id
        file_to_case = {}
        for file_name in sample_files:
            metadata_row = df[df['keypoint_filename'] == file_name]
            if not metadata_row.empty:
                case_id = metadata_row['case_id'].iloc[0]
                file_to_case[file_name] = case_id

        # Statistics per label
        label_stats = defaultdict(lambda: {'case_ids': set(), 'segments': 0})

        for file_name, label in zip(sample_files, sample_labels):
            if file_name in file_to_case:
                case_id = file_to_case[file_name]
                label_stats[label]['case_ids'].add(case_id)
                label_stats[label]['segments'] += 1

        # Convert to arrays and log
        stats_array = []
        for label in sorted(label_stats.keys()):
            num_case_ids = len(label_stats[label]['case_ids'])
            num_segments = label_stats[label]['segments']
            stats_array.append([label, num_case_ids, num_segments])

        logging.info(f"{phase} label statistics: {stats_array}")
