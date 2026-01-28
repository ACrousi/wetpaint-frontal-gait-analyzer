import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def main():
    # 設定範圍
    range_min = 12
    range_max = 36
    x = np.arange(range_min, range_max + 1)

    # 初始參數
    init_mu = 24.0
    init_sigma = 1.0
    init_beta = 2.0  # Gaussian when beta=2

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, bottom=0.35)

    # 初始繪圖
    bars = ax.bar(x, np.zeros_like(x), align='center', alpha=0.7, color='skyblue', edgecolor='black')
    line, = ax.plot(x, np.zeros_like(x), 'r--', alpha=0.5)
    
    ax.set_xlim(range_min - 1, range_max + 1)
    ax.set_ylim(0, 1.1)  # 機率不會超過 1
    ax.set_xlabel('Value (Integer)')
    ax.set_ylabel('Probability')
    ax.set_title(f'Generalized Normal Distribution (Range: {range_min}-{range_max})')
    ax.set_xticks(x)
    
    # 顯示數值標籤的函式
    def autolabel(rects, probs):
        for rect, prob in zip(rects, probs):
            height = rect.get_height()
            # 這裡我們不直接用 height (因為會變動), 而是用 prob
            # 但 Bar chart update 時 height 會變
            pass 

    # 更新圖表的函式
    def update(val):
        mu = s_mu.val
        sigma = s_sigma.val
        beta = s_beta.val

        # 計算 Generalized Normal Distribution (GND)
        # 參考 ResGCNv1 coco.py 的寫法並推廣: np.exp(-0.5 * (diff / sigma) ** 2)
        # 這裡從 2 改為 beta: np.exp(-0.5 * (abs(diff) / sigma) ** beta)
        # 注意: 加上 0.5 係數是為了在 Beta=2 時與 Gaussian (標準差 sigma) 一致
        diff = np.abs(x - mu)
        dist = np.exp(-0.5 * (diff / sigma) ** beta)
        
        # 正規化
        dist_sum = dist.sum()
        if dist_sum > 0:
            dist /= dist_sum
        
        # 更新 Bar 高度
        max_height = 0
        for bar, h in zip(bars, dist):
            bar.set_height(h)
            if h > max_height:
                max_height = h
        
        # 更新 Line
        line.set_ydata(dist)
        
        # 更新 Y 軸範圍 (隨著分布變窄，峰值會變高)
        ax.set_ylim(0, max(1.1 * max_height, 0.1))
        
        fig.canvas.draw_idle()

    # 設置 Sliders
    axcolor = 'lightgoldenrodyellow'
    
    # Target / Mean Slider
    ax_mu = plt.axes([0.2, 0.2, 0.65, 0.03], facecolor=axcolor)
    s_mu = Slider(ax_mu, 'Target (Mean)', range_min, range_max, valinit=init_mu, valstep=1.0)
    
    # Sigma Slider (寬度)
    ax_sigma = plt.axes([0.2, 0.15, 0.65, 0.03], facecolor=axcolor)
    s_sigma = Slider(ax_sigma, 'Sigma (Std)', 0.1, 10.0, valinit=init_sigma, valstep=0.1)
    
    # Beta Slider (峰值形狀)
    ax_beta = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
    s_beta = Slider(ax_beta, 'Beta (Shape)', 0.5, 10.0, valinit=init_beta, valstep=0.1)

    # 綁定更新事件
    s_mu.on_changed(update)
    s_sigma.on_changed(update)
    s_beta.on_changed(update)

    # Reset 按鈕
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    
    def reset(event):
        s_mu.reset()
        s_sigma.reset()
        s_beta.reset()
    button.on_clicked(reset)

    # 初始化一次
    update(None)

    plt.show()

if __name__ == "__main__":
    main()
