import pickle
import logging
import numpy as np
from torch.utils.data import Dataset

from .data_utils import multi_input
from .skeleton_transform import SkeletonTransform

logger = logging.getLogger()


class SkeletonAugmentor:
    """
    Data augmentation for skeleton sequences.
    Config-driven augmentation suitable for infant gait analysis.
    
    Safe augmentations that preserve gait features:
    - Translation: Shifts position, doesn't affect movement patterns
    - Scale: Simulates distance variation, preserves proportions
    - Gaussian Noise: Simulates pose estimation errors
    """
    
    def __init__(self, config=None):
        """
        Args:
            config: dict with augmentation parameters
                - enabled: bool, whether to apply augmentation
                - translation_range: float, max translation (default: 0.1)
                - scale_range: [min, max], scale range (default: [0.95, 1.05])
                - noise_std: float, Gaussian noise std (default: 0.005)
                - translation_prob: float, probability of applying translation (default: 0.5)
                - scale_prob: float, probability of applying scale (default: 0.5)
                - noise_prob: float, probability of applying noise (default: 0.5)
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', False)
        
        # Augmentation parameters with safe defaults
        self.translation_range = self.config.get('translation_range', 0.1)
        self.scale_range = self.config.get('scale_range', [0.95, 1.05])
        self.noise_std = self.config.get('noise_std', 0.005)
        
        # Probability of applying each augmentation
        self.translation_prob = self.config.get('translation_prob', 0.5)
        self.scale_prob = self.config.get('scale_prob', 0.5)
        self.noise_prob = self.config.get('noise_prob', 0.5)
        
        if self.enabled:
            logging.info(f"SkeletonAugmentor enabled: translation={self.translation_range}, "
                        f"scale={self.scale_range}, noise_std={self.noise_std}")
    
    def __call__(self, data):
        """
        Apply augmentation to skeleton data.
        
        Args:
            data: numpy array of shape (C, T, V, M) where
                  C=channels (x,y,z,...), T=frames, V=joints, M=persons
        
        Returns:
            Augmented data with same shape
        """
        if not self.enabled:
            return data
        
        data = data.copy()  # Avoid modifying original data
        C, T, V, M = data.shape
        
        # 1. Random Translation (平移) - preserves all movement patterns
        if np.random.rand() < self.translation_prob:
            # Apply same shift to all frames (global position change)
            shift = np.random.uniform(-self.translation_range, self.translation_range, (C, 1, 1, 1))
            data = data + shift
        
        # 2. Random Scale (縮放) - preserves proportions and patterns
        if np.random.rand() < self.scale_prob:
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            data = data * scale
        
        # 3. Gaussian Noise (噪聲) - simulates pose estimation error
        if np.random.rand() < self.noise_prob:
            noise = np.random.normal(0, self.noise_std, data.shape)
            data = data + noise
        
        return data.astype(np.float32)


class Feeder(Dataset):
    def __init__(self, phase, path=None, data_shape=None, connect_joint=None, debug=False,
                 data_path=None, label_path=None, eval_data_path=None, eval_label_path=None,
                 gait_path=None, use_gait=False, model_type='resgcn', is_regression=False,
                 use_ldl=False, ldl_config=None, num_class=1, ldl_min_label=None,
                 ldl_max_label=None, custom_bin_centers=None, augmentation=None, **kwargs):
        self.split = phase
        self.conn = connect_joint if connect_joint is not None else []
        self.use_gait = use_gait
        self.model_type = model_type
        self.is_regression = is_regression
        self.use_ldl = use_ldl
        self.ldl_config = ldl_config if ldl_config is not None else {'sigma': 1.5, 'beta': 2.0}
        self.num_class = num_class
        self.ldl_min_label = ldl_min_label
        self.ldl_max_label = ldl_max_label
        self.custom_bin_centers = custom_bin_centers  # 支援非均勻 bin centers
        
        # Initialize augmentor (only for training phase)
        if phase == 'train' and augmentation is not None:
            augmentor = SkeletonAugmentor(augmentation)
        else:
            augmentor = SkeletonAugmentor({'enabled': False})
        
        # Initialize skeleton transform (shared preprocessing logic)
        self.skeleton_transform = SkeletonTransform(
            connect_joint=connect_joint if connect_joint is not None else [],
            augmentor=augmentor
        )

        if phase == 'train':
            self.data_path = data_path or f"{path}/train_data.npy"
            self.label_path = label_path or f"{path}/train_label.pkl"
            self.gait_path = gait_path or f"{path}/train_gait.npy" if use_gait else None
        elif phase == 'eval':
            self.data_path = eval_data_path or f"{path}/val_data.npy"
            self.label_path = eval_label_path or f"{path}/val_label.pkl"
            self.gait_path = gait_path or f"{path}/val_gait.npy" if use_gait else None
        else:
            raise ValueError(f"Invalid phase: {phase}")

        self.load_data()

    def load_data(self):
        # 載入標籤
        try:
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

        # 資料驗證：檢查標籤是否有 NaN 或無效值
        self.label = np.array(self.label, dtype=np.float32)
        if np.any(np.isnan(self.label)) or np.any(np.isinf(self.label)):
            logging.warning(f"Found NaN or Inf in labels from {self.label_path}")
            # 將 NaN 和 Inf 替換為 0
            self.label = np.nan_to_num(self.label, nan=0.0, posinf=0.0, neginf=0.0)

        logging.info(f"Loaded {len(self.label)} labels, range: {self.label.min():.2f} - {self.label.max():.2f}")

        # 初始化 bin centers（在載入標籤後填入）
        self.bin_centers = None

        if self.use_ldl:
            self._prepare_bins()

        # 根據模型類型載入資料
        if self.model_type == 'gaitmlp':
            # 僅載入 gait
            self.data = None
            if self.gait_path:
                self.gait_data = np.load(self.gait_path)
            else:
                raise ValueError("gaitmlp requires gait data")
        else:
            # 載入 frames
            self.data = np.load(self.data_path)

            # 如果需要，載入 gait
            if self.use_gait and self.gait_path:
                self.gait_data = np.load(self.gait_path)
            else:
                self.gait_data = None

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        label = self.label[index]  # 原始標籤值
        name = self.sample_name[index] if hasattr(self, 'sample_name') else str(index)

        # 如果使用 LDL，將標籤轉換為分佈
        if self.use_ldl:
            label_dist = self._label_to_distribution(label)
            original_label = label  # 保留原始標籤用於 MAE 計算
        else:
            # Regression or classification: use raw label as float
            label_dist = np.float32(label)
            original_label = label

        if self.model_type == 'gaitmlp':
            # 模式 1：僅返回 gait features
            gait_params = self.gait_data[index]
            # 返回 dummy data（因為 GaitMLP 只用 gait_params）
            dummy_data = np.zeros((3, 6, 150, 17, 1), dtype=np.float32)
            return dummy_data, label_dist, original_label, gait_params, name
        else:
            # 模式 2 & 3：返回 frames
            data_numpy = np.array(self.data[index])
            
            # 使用 SkeletonTransform 進行資料預處理
            # (joint masking + augmentation + multi_input)
            data_numpy = self.skeleton_transform(data_numpy)

            # 獲取 gait（如果有）
            gait_params = self.gait_data[index] if self.gait_data is not None else None

            if gait_params is not None:
                return data_numpy, label_dist, original_label, gait_params, name
            else:
                return data_numpy, label_dist, original_label, name

    def _prepare_bins(self):
        if self.bin_centers is not None:
            return

        # 優先使用自定義 bin centers（非均勻分布）
        if self.custom_bin_centers is not None:
            self.bin_centers = np.array(self.custom_bin_centers, dtype=np.float32)
            self.num_class = len(self.bin_centers)
            logging.info(f"Using custom bin centers: {self.bin_centers} ({self.num_class} classes)")
            return

        # 回退到均勻 bin centers
        label_min = self.ldl_min_label if self.ldl_min_label is not None else float(self.label.min())
        label_max = self.ldl_max_label if self.ldl_max_label is not None else float(self.label.max())

        if self.num_class <= 1:
            self.num_class = int(round(label_max - label_min)) + 1

        self.bin_centers = np.linspace(label_min, label_max, self.num_class, dtype=np.float32)
        logging.info(f"Using uniform bin centers: {self.bin_centers[0]:.1f} to {self.bin_centers[-1]:.1f} ({self.num_class} classes)")

    def _label_to_distribution(self, label_value):
        """
        Convert label to distribution based on configuration.
        
        LDL types:
        - 'gaussian': Gaussian distribution in index space (default)
        - 'ordinal': Ordinal cumulative distribution
        - 'exact': Linear interpolation ensuring E[dist] = label_value exactly
        - 'exact_soft': Soft version of exact with Gaussian smoothing
        """
        # Determine LDL type
        ldl_type = 'gaussian'
        if isinstance(self.ldl_config, dict):
            ldl_type = self.ldl_config.get('type', 'gaussian')
        
        if ldl_type == 'ordinal':
            return self._label_to_distribution_ordinal(label_value)
        elif ldl_type == 'exact':
            return self._label_to_distribution_exact(label_value)
        elif ldl_type == 'exact_soft':
            return self._label_to_distribution_exact_soft(label_value)
        else:
            return self._label_to_distribution_gaussian(label_value)

    def _label_to_distribution_ordinal(self, label_value):
        """OLDL: Use cumulative distribution method via sigmoid"""
        self._prepare_bins()
        
        # Determine scale parameter
        if isinstance(self.ldl_config, dict):
            scale = float(self.ldl_config.get('ordinal_scale', 1.0))
        else:
            scale = 1.0

        # Number of bins
        num_bins = len(self.bin_centers)
        bin_indices = np.arange(num_bins, dtype=np.float32)
        
        # Find continuous index of the label
        target_idx = np.interp(label_value, self.bin_centers, bin_indices)
        
        # Calculate cumulative probability: P(Y > k) = sigmoid((target - k) / scale)
        # Note: Usually Ordinal Regression uses P(y > k) = sigmoid(score_k). 
        # Here we are generating the ground truth distribution.
        # We model the cumulative distribution P(Y <= k) as sigmoid((k - target) / scale).
        # Or more standard for "soft" ordinal labels: 
        # For each rank k, we want probability mass centered around target.
        # But for OLDL using cumulative loss, we often just want the CDF.
        # Let's generate a PDF that respects ordinality (unimodal) and then let loss handle CDF.
        # Actually, standard OLDL often uses the method where we generate soft labels representing P(y > k).
        
        # Approach: Generate PDF using the simple cumulative difference method:
        # P(Y <= k) = sigmoid((k - target_idx) / scale)
        # Prob(k) = P(Y <= k) - P(Y <= k-1)
        
        # Calculate CDF values at bin boundaries (conceptual boundaries between k and k+1)
        # Let's define CDF at index k as P(Y <= k).
        # To make it centered at target_idx, sigmoid((k - target_idx) / scale) works.
        
        probs_cumulative = 1.0 / (1.0 + np.exp(-(bin_indices - target_idx) / scale))
        
        # Convert cumulative probabilities to PDF (discrete)
        # dist[0] = P(Y <= 0)
        # dist[k] = P(Y <= k) - P(Y <= k-1) for k > 0
        
        dist = np.zeros(num_bins, dtype=np.float32)
        dist[0] = probs_cumulative[0]
        dist[1:] = probs_cumulative[1:] - probs_cumulative[:-1]
        
        # Normalize just in case, though it should be close to 1
        dist = np.clip(dist, 0, 1) # Ensure non-negative
        dist_sum = dist.sum()
        if dist_sum > 0:
            dist /= dist_sum
            
        return dist

    def _label_to_distribution_gaussian(self, label_value):
        """Original Gaussian LDL"""
        self._prepare_bins()
        
        if isinstance(self.ldl_config, dict):
            sigma = float(self.ldl_config.get('sigma', 1.5))
            beta = float(self.ldl_config.get('beta', 2.0))
        else:
            sigma = 1.5
            beta = 2.0

        bin_indices = np.arange(len(self.bin_centers), dtype=np.float32)
        continuous_idx = np.interp(label_value, self.bin_centers, bin_indices)
        
        idx_diff = bin_indices - continuous_idx
        
        dist = np.exp(-0.5 * (idx_diff / sigma) ** beta)
        dist = dist.astype(np.float32)
        dist_sum = dist.sum()
        if dist_sum > 0:
            dist /= dist_sum
        return dist

    def _label_to_distribution_exact(self, label_value):
        """
        Exact LDL: Use linear interpolation to ensure E[distribution] = label_value exactly.
        
        This method assigns probability only to the two adjacent bins such that
        the expectation value equals the original label precisely.
        
        For label L between bin[i] and bin[i+1]:
            P(bin[i]) = (bin[i+1] - L) / (bin[i+1] - bin[i])
            P(bin[i+1]) = (L - bin[i]) / (bin[i+1] - bin[i])
        
        This guarantees: E[dist] = P[i] * bin[i] + P[i+1] * bin[i+1] = L
        """
        self._prepare_bins()
        
        num_bins = len(self.bin_centers)
        dist = np.zeros(num_bins, dtype=np.float32)
        
        # Clamp label to valid range
        label_clamped = np.clip(label_value, self.bin_centers[0], self.bin_centers[-1])
        
        # Find which two bins the label falls between
        if label_clamped <= self.bin_centers[0]:
            # At or below minimum bin
            dist[0] = 1.0
        elif label_clamped >= self.bin_centers[-1]:
            # At or above maximum bin
            dist[-1] = 1.0
        else:
            # Find the bin interval [bin[i], bin[i+1]] containing the label
            for i in range(num_bins - 1):
                if self.bin_centers[i] <= label_clamped <= self.bin_centers[i + 1]:
                    bin_low = self.bin_centers[i]
                    bin_high = self.bin_centers[i + 1]
                    
                    # Linear interpolation weights
                    w_high = (label_clamped - bin_low) / (bin_high - bin_low)
                    w_low = 1.0 - w_high
                    
                    dist[i] = w_low
                    dist[i + 1] = w_high
                    break
        
        return dist

    def _label_to_distribution_exact_soft(self, label_value):
        """
        Exact Soft LDL: Gaussian-smoothed distribution that still maintains E[dist] ≈ label_value.
        
        This method:
        1. Creates a Gaussian distribution centered on the label (in value space)
        2. Adjusts the distribution to ensure the expectation matches the original label
        
        The adjustment is done by iteratively shifting the center to correct any expectation error.
        """
        self._prepare_bins()
        
        if isinstance(self.ldl_config, dict):
            sigma = float(self.ldl_config.get('sigma', 1.5))
            max_iterations = int(self.ldl_config.get('exact_iterations', 10))
        else:
            sigma = 1.5
            max_iterations = 10
        
        num_bins = len(self.bin_centers)
        
        # Clamp label to valid range
        label_clamped = np.clip(label_value, self.bin_centers[0], self.bin_centers[-1])
        
        # Calculate bin spacing (average for non-uniform bins)
        avg_spacing = (self.bin_centers[-1] - self.bin_centers[0]) / (num_bins - 1)
        sigma_value = sigma * avg_spacing  # Convert sigma from index space to value space
        
        # Initial Gaussian centered at the label (in value space, not index space)
        target_center = label_clamped
        
        for iteration in range(max_iterations):
            # Calculate Gaussian distribution in value space
            value_diff = self.bin_centers - target_center
            dist = np.exp(-0.5 * (value_diff / sigma_value) ** 2)
            dist = dist.astype(np.float32)
            
            # Normalize
            dist_sum = dist.sum()
            if dist_sum > 0:
                dist /= dist_sum
            
            # Calculate current expectation
            current_exp = np.sum(dist * self.bin_centers)
            
            # Check if expectation matches
            error = label_clamped - current_exp
            if abs(error) < 0.001:  # Tolerance of 0.001
                break
            
            # Adjust center to correct the error
            target_center += error * 0.5  # Damped correction
            
            # Clamp target_center to valid range
            target_center = np.clip(target_center, self.bin_centers[0], self.bin_centers[-1])
        
        return dist