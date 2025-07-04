import csv
from ollama import chat

LIMIT = 10
MODEL = 'gemma3:12b-it-qat'
INPUT_FILE = "posts_groundtruth.txt"
OUTPUT_FILE = "sentiment_predictions.csv"

# 加载数据集
def load_dataset(file_path, limit=None):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            dataset.append({
                "post_id": row["post_id"],
                "post_text": row["post_text"]
            })
    return dataset

# 情感分析函数
def analyze_sentiment(post_text):
    prompt = f"""
你是一个 情感识别专家，请判断下面这段社交媒体内容的情感倾向，并严格按照以下格式回复：

内容：
{post_text}

请只回复一行：
情感：积极 / 中性 / 消极（仅保留一个，不要添加其它文字）
""".strip()

    try:
        response = chat(model=MODEL, messages=[{
            'role': 'user',
            'content': prompt
        }])
        reply = response['message']['content'].strip()

        # 标准化输出
        if "积极" in reply:
            return "positive"
        elif "中性" in reply:
            return "neutral"
        elif "消极" in reply:
            return "negative"
        else:
            return "unknown"
    except Exception as e:
        print(f"[Sentiment Error] {e}")
        return "error"

# 主函数
def run_sentiment_analysis(dataset, save_path=OUTPUT_FILE):
    results = []

    for i, entry in enumerate(dataset):
        sentiment = analyze_sentiment(entry["post_text"])
        print(f"[{i+1}/{len(dataset)}] Sentiment: {sentiment}")
        results.append({
            "post_id": entry["post_id"],
            "post_text": entry["post_text"],
            "sentiment": sentiment
        })

    # 写入CSV
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["post_id", "post_text", "sentiment"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ 情感分析完成，结果已保存至：{save_path}")

# 执行
if __name__ == "__main__":
    data = load_dataset(INPUT_FILE, limit=LIMIT)
    run_sentiment_analysis(data)
