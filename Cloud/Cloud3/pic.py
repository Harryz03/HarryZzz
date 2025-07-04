import matplotlib.pyplot as plt
import numpy as np

# 指标数据
categories = ['Fake News (0)', 'Real News (1)']
precision = [0.7143, 0.5652]
recall = [0.3333, 0.8667]
f1_score = [0.4545, 0.6842]
accuracy = 0.6

x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

# 绘制三组指标柱状图
bar1 = ax.bar(x - width, precision, width, label='Precision', color='#1f77b4')
bar2 = ax.bar(x, recall, width, label='Recall', color='#ff7f0e')
bar3 = ax.bar(x + width, f1_score, width, label='F1 Score', color='#2ca02c')

# 添加准确率文本（单独展示）
ax.text(0.5, 0.95, f'Overall Accuracy: {accuracy:.2f}', transform=ax.transAxes,
        fontsize=14, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='lightgray', alpha=0.5))

# 设置标签和标题
ax.set_ylabel('Scores')
ax.set_title('Performance Metrics for Fake News Detection')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0, 1)

# 添加图例
ax.legend()

# 给柱子添加数值标签
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bar1)
add_labels(bar2)
add_labels(bar3)

plt.tight_layout()
plt.show()
