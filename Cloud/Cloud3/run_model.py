from transformers import BertTokenizer
from torchvision import transforms
from PIL import Image
import torch
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from eric_fnd_model import ERICFND

# === 初始化模型和预处理 ===
model = ERICFND()
model.eval()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === 读取数据 ===
with open("news_dataset.json", "r", encoding="utf-8") as f:
    news_data = json.load(f)[:30]

with open("entity_knowledge.json", "r", encoding="utf-8") as f:
    entity_knowledge = json.load(f)

# === 预测及评估准备 ===
y_true = []
y_pred = []

for idx, item in enumerate(news_data):
    text = item["text"]
    image_path = item["image_path"]
    label = item["label"]

    # 文本编码
    encoded = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=256)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # 图像处理
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    # 实体处理（示例用零张量）
    entity_tensor = torch.zeros((1, 1, 768))

    # 模型推理
    with torch.no_grad():
        output, _, _ = model(input_ids, attention_mask, image_tensor, entity_tensor)
        prediction = torch.argmax(output, dim=1).item()

    y_true.append(label)
    y_pred.append(prediction)

    print(f"[{idx + 1:02d}] ✅ 预测: {prediction} | 实际: {label}")

# 计算整体准确率
accuracy = accuracy_score(y_true, y_pred)

# 计算两类的precision, recall, f1
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0,1])

print(f"\n整体准确率 Accuracy: {accuracy:.4f}")
print(f"假新闻 (label=0) Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1-score: {f1[0]:.4f}, Support: {support[0]}")
print(f"真新闻 (label=1) Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1-score: {f1[1]:.4f}, Support: {support[1]}")
