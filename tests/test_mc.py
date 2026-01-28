import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 根據您提供的數據定義混淆矩陣
confusion_matrix_data = np.array([
    [6, 1, 1],
    [0, 9, 3],
    [0, 1, 9]
])

# 定義X軸與Y軸的標籤
x_axis_labels = ["12-18", "18-24", "24-36"]
y_axis_labels = ["12-18", "18-24", "24-36"]

# 建立熱圖 (heatmap)
plt.figure(figsize=(8, 6))
sns.heatmap(
    confusion_matrix_data,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=x_axis_labels,
    yticklabels=y_axis_labels,
    annot_kws={"size": 16}
)

# 加入標題與座標軸名稱
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Month Prediction Confusion Matrix", fontsize=16)

# 將圖表儲存為圖片檔案
plt.savefig("confusion_matrix.png")