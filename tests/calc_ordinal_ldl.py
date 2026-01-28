import numpy as np

# LDL 參數
bin_centers = np.array([12, 18, 24, 30, 36], dtype=np.float32)

# Ordinal LDL 參數
ordinal_scale = 0.2

# Gaussian LDL 參數
gaussian_sigma = 1.0
gaussian_beta = 2.0


def label_to_distribution_ordinal(label_value, bin_centers, scale):
    """Ordinal LDL: 使用累積 sigmoid 方法"""
    num_bins = len(bin_centers)
    bin_indices = np.arange(num_bins, dtype=np.float32)
    
    # 找到標籤的連續索引
    target_idx = np.interp(label_value, bin_centers, bin_indices)
    
    # 計算累積機率: P(Y <= k) = sigmoid((k - target_idx) / scale)
    probs_cumulative = 1.0 / (1.0 + np.exp(-(bin_indices - target_idx) / scale))
    
    # 轉換為 PDF
    dist = np.zeros(num_bins, dtype=np.float32)
    dist[0] = probs_cumulative[0]
    dist[1:] = probs_cumulative[1:] - probs_cumulative[:-1]
    
    # 正規化
    dist = np.clip(dist, 0, 1)
    dist_sum = dist.sum()
    if dist_sum > 0:
        dist /= dist_sum
        
    return dist, target_idx


def label_to_distribution_gaussian(label_value, bin_centers, sigma, beta):
    """Gaussian LDL: 使用廣義高斯分布"""
    num_bins = len(bin_centers)
    bin_indices = np.arange(num_bins, dtype=np.float32)
    
    # 找到標籤的連續索引
    continuous_idx = np.interp(label_value, bin_centers, bin_indices)
    
    # 計算廣義高斯分布
    idx_diff = bin_indices - continuous_idx
    dist = np.exp(-0.5 * (idx_diff / sigma) ** beta)
    
    # 正規化
    dist = dist.astype(np.float32)
    dist_sum = dist.sum()
    if dist_sum > 0:
        dist /= dist_sum
        
    return dist, continuous_idx


def distribution_to_expectation(dist, bin_centers):
    """計算分布的期望值"""
    return np.sum(dist * bin_centers)


def print_analysis(mode, bin_centers, **params):
    """打印分析結果"""
    if mode == 'ordinal':
        scale = params.get('scale', 1.0)
        print('='*90)
        print(f'Ordinal LDL 標籤分布期望值分析')
        print('='*90)
        print(f'Bin Centers: {bin_centers}')
        print(f'Ordinal Scale: {scale}')
        print('='*90)
        
        dist_func = lambda label: label_to_distribution_ordinal(label, bin_centers, scale)
        
    elif mode == 'gaussian':
        sigma = params.get('sigma', 1.5)
        beta = params.get('beta', 2.0)
        print('='*90)
        print(f'Gaussian LDL 標籤分布期望值分析')
        print('='*90)
        print(f'Bin Centers: {bin_centers}')
        print(f'Sigma: {sigma}, Beta: {beta}')
        print('='*90)
        
        dist_func = lambda label: label_to_distribution_gaussian(label, bin_centers, sigma, beta)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    print(f'\n原始標籤 | 連續索引 | 分布機率 (12, 18, 24, 30, 36)             | 期望值  | 偏差')
    print('-'*90)
    
    for label in range(12, 37):  # 12 到 36
        dist, cont_idx = dist_func(label)
        expectation = distribution_to_expectation(dist, bin_centers)
        bias = expectation - label
        
        dist_str = ' '.join([f'{p:.4f}' for p in dist])
        marker = ' <-- bin center' if label in bin_centers else ''
        print(f'{label:^8} | {cont_idx:^8.3f} | [{dist_str}] | {expectation:6.2f} | {bias:+5.2f}{marker}')
    
    print('='*90)


if __name__ == '__main__':
    # ============================================
    # Ordinal LDL 分析
    # ============================================
    print("\n" + "="*90)
    print(" " * 30 + "【 Ordinal LDL 】")
    print("="*90)
    print_analysis('ordinal', bin_centers, scale=ordinal_scale)
    
    print("\n關鍵觀察 (Ordinal LDL):")
    print("- 使用累積 sigmoid，分布呈現不對稱")
    print("- 邊界標籤偏差較大（無法被兩側平衡）")
    print("- scale 越小，分布越集中在最近的 bin")
    
    # ============================================
    # Gaussian LDL 分析
    # ============================================
    print("\n\n" + "="*90)
    print(" " * 30 + "【 Gaussian LDL 】")
    print("="*90)
    print_analysis('gaussian', bin_centers, sigma=gaussian_sigma, beta=gaussian_beta)
    
    print("\n關鍵觀察 (Gaussian LDL):")
    print("- 使用對稱的廣義高斯分布")
    print("- 邊界標籤也會有偏差（因為分布被截斷）")
    print("- sigma 控制分布寬度，beta 控制尖峰程度")
    
    # ============================================
    # 兩種方法比較
    # ============================================
    print("\n\n" + "="*90)
    print(" " * 30 + "【 兩種方法比較 】")
    print("="*90)
    print(f"Ordinal Scale: {ordinal_scale}, Gaussian Sigma: {gaussian_sigma}, Beta: {gaussian_beta}")
    print('-'*90)
    print(f'原始標籤 | Ordinal期望值 | Ordinal偏差 | Gaussian期望值 | Gaussian偏差')
    print('-'*90)
    
    for label in range(12, 37):
        dist_ord, _ = label_to_distribution_ordinal(label, bin_centers, ordinal_scale)
        exp_ord = distribution_to_expectation(dist_ord, bin_centers)
        bias_ord = exp_ord - label
        
        dist_gauss, _ = label_to_distribution_gaussian(label, bin_centers, gaussian_sigma, gaussian_beta)
        exp_gauss = distribution_to_expectation(dist_gauss, bin_centers)
        bias_gauss = exp_gauss - label
        
        marker = ' <-- bin center' if label in bin_centers else ''
        print(f'{label:^8} | {exp_ord:^13.2f} | {bias_ord:^+11.2f} | {exp_gauss:^14.2f} | {bias_gauss:^+12.2f}{marker}')
    
    print('='*90)
