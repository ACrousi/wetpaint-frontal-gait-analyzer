import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr

# 讀取CSV文件
df = pd.read_csv('outputs/csv/track_analysis_metadata.csv')
df = df.drop_duplicates(subset='video_name', keep='first')
# df = df[['age_week', 'spine_angle_median', 'knee_distance_median', 'ankle_distance_median', 'developmental_retardation']]
# sns.pairplot(df, diag_kind='kde', corner=True)
# plt.show()

# df = df[(df['age_months'] > 18) & (df['age_months'] < 24)]
# df = df[(df['age_months'] < 18)]
# 假設月齡是 'age_months', 目標欄位是 'knee_distance_median', label是 'age_months_range'
x = df['age_week']
y_name = 'knee_distance_std'
y = df[y_name]
# y = df['spine_angle_max'] - df['spine_angle_min']
# labels = pd.Categorical(df['age_months_range'].map({0: "12-15", 1: "15-18", 2: "18-24", 3: "24-30", 4: "30~"}), categories=["12-15", "15-18", "18-24", "24-30", "30~"], ordered=True)
labels = pd.Categorical(df['developmental_retardation'].map({0: "normal", 1: "Slight", 2: "Severe"}), categories=["normal", "Slight", "Severe"], ordered=True)

# 計算相關係數
corr, _ = pearsonr(x, y)
print(f"Correlation coefficient between age_week and {y_name}: {corr}")

# 繪製散點圖
plt.figure(figsize=(10, 6))
sns.scatterplot(x=x, y=y, hue=labels, palette='tab10', s=50)
# sns.scatterplot(x=x, y=y, color='gray', s=50, legend=False)

# # 添加id標籤
# for i in range(len(df)):
#     plt.text(x.iloc[i], y.iloc[i], str(df['video_name'].iloc[i])+' '+str(df['track_id'].iloc[i]), fontsize=8, ha='right', va='bottom')

# 添加趨勢線
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--", label=f'Trend line (corr={corr:.2f})')

# 設置標籤和標題
plt.xlabel('weeks')
plt.ylabel(y_name)
plt.title(f'weeks vs {y_name}')
plt.legend()
plt.grid(True)

# 保存圖表
plt.savefig('plot_label.png', dpi=300, bbox_inches='tight')
print("圖表已保存為 plot_label.png")

# 顯示圖表
plt.show()