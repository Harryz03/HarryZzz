import csv
from ollama import chat

LIMIT = 1000
INPUT_FILE = "posts_groundtruth.txt"
OUTPUT_FILE = "result_with_sentiment.csv"

def load_dataset(file_path, limit=None):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            dataset.append({
                "post_id": row["post_id"],
                "post_text": row["post_text"],
                "label": row["label"].strip().lower()
            })
    return dataset

def analyze_sentiment(text):
    prompt = f"""
请你判断下面这段社交媒体内容的情感倾向，只能回答“积极”、“中性”或“消极”三种之一，不要输出解释：

内容：
{text}
"""
    try:
        response = chat(model='gemma3:12b-it-qat', messages=[{
            'role': 'user',
            'content': prompt.strip()
        }])
        reply = response['message']['content'].strip()
        if "积极" in reply:
            return "积极"
        elif "消极" in reply:
            return "消极"
        elif "中性" in reply:
            return "中性"
        else:
            return "未知"
    except Exception as e:
        print(f"[情感分析错误] {e}")
        return "error"

def predict_fakeness(text, sentiment=None):
    prompt = f"""
你是一个新闻真伪判别专家。请根据下面的内容判断它是“真新闻”还是“假新闻”。你只能回答“真”或“假”，不能输出其它内容或解释。

请注意：
1、“假新闻”指的是虚构、歪曲或误导性的信息，
2、“真新闻”指的是基于事实、可以验证的信息。
3、请根据情感分析结果与新闻内容本身进行判断。

情感分析结果：{sentiment}

现在请判断以下内容：

{text}
"""
    try:
        response = chat(model='gemma3:12b-it-qat', messages=[{
            'role': 'user',
            'content': prompt.strip()
        }])
        reply = response['message']['content'].strip()
        print(f"模型回复: {reply}")
        if reply == "真":
            return "real"
        elif reply == "假":
            return "fake"
        else:
            return "unknown"
    except Exception as e:
        print(f"[真假判断错误] {e}")
        return "error"

def run_analysis(data, save_path=OUTPUT_FILE):
    results = []

    for i, entry in enumerate(data):
        post_id = entry["post_id"]
        text = entry["post_text"]
        true_label = entry["label"]

        sentiment = analyze_sentiment(text)
        pred_label = predict_fakeness(text, sentiment=sentiment)

        is_correct = pred_label == true_label
        symbol = "✅" if is_correct else "❌"
        print(f"[{i+1}/{len(data)}] Pred: {pred_label}, True: {true_label} Sentiment: {sentiment} {symbol}")

        results.append({
            "post_id": post_id,
            "post_text": text,
            "true_label": true_label,
            "predicted": pred_label,
            "correct": is_correct,
            "sentiment": sentiment
        })

    correct = sum(1 for r in results if r['correct'])
    total = len(results)

    fake_total = sum(1 for r in results if r['true_label'] == 'fake')
    fake_correct = sum(1 for r in results if r['true_label'] == 'fake' and r['predicted'] == 'fake')

    real_total = sum(1 for r in results if r['true_label'] == 'real')
    real_correct = sum(1 for r in results if r['true_label'] == 'real' and r['predicted'] == 'real')

    acc = correct / total * 100
    acc_fake = fake_correct / fake_total * 100 if fake_total else 0
    acc_real = real_correct / real_total * 100 if real_total else 0

    print("\n📊 准确率统计结果：")
    print(f"✅ Accuracy       : {acc:.2f}%")
    print(f"🧪 Accuracy_fake  : {acc_fake:.2f}%")
    print(f"📢 Accuracy_true  : {acc_real:.2f}%")

    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "post_id", "post_text", "true_label", "predicted", "correct", "sentiment"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"📄 已保存带情感分析的结果到: {save_path}")

if __name__ == "__main__":
    dataset = load_dataset(INPUT_FILE, limit=LIMIT)
    run_analysis(dataset)
