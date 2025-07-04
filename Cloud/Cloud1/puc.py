import matplotlib.pyplot as plt

labels = ['Accuracy', 'Accuracy_fake', 'Accuracy_true']
no_sentiment = [60.30, 75.80, 50.96]
with_sentiment = [65.60, 60.37, 68.75]

x = range(len(labels))
width = 0.35

plt.figure(figsize=(10,6))
plt.bar([i - width/2 for i in x], no_sentiment, width=width, label='No Sensitive', color='#6baed6')
plt.bar([i + width/2 for i in x], with_sentiment, width=width, label='With Sensitive', color='#fd8d3c')

plt.xticks(x, labels, fontsize=12)
plt.ylabel('Accuracy（%）', fontsize=13)
plt.title('Impact of Sentiment Analysis on Fake News Detection Accuracy', fontsize=14)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
