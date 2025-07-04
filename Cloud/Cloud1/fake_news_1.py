import csv
from ollama import chat

#对数据集的读取数量进行限制
LIMIT = 1000
INPUT_FILE = "posts_groundtruth.txt"
OUTPUT_FILE = "result_no_sentiment.csv"

#读取数据，并提取出其中有用的部分
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
#prompt
def predict_fakeness(text):
    prompt = f"""
你是一个新闻真伪判别专家。请根据下面的内容判断它是“真新闻”还是“假新闻”。你只能回答“真”或“假”，不能输出其它内容或解释。

请注意：“假新闻”指的是虚构、歪曲或误导性的信息，而“真新闻”指的是基于事实、可以验证的信息。

现在请判断以下内容：

内容：
"{text}"

"""
#匹配大模型判断，并输出大模型判断结果
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
        print(f"[Error] {e}")
        return "error"


#将大模型的运行输出结果与数据集中的真假进行匹配，判断大模型的回答是否正确
def run_analysis(data, save_path=OUTPUT_FILE, save_prefix='no_sentiment'):
    results = []

    for i, entry in enumerate(data):
        post_id = entry["post_id"]
        text = entry["post_text"]
        true_label = entry["label"]

        pred_label = predict_fakeness(text)
        is_correct = pred_label == true_label
        symbol = "✅" if is_correct else "❌"
        print(f"[{i+1}/{len(data)}] Pred: {pred_label}, True: {true_label} {symbol}")

        results.append({
            "post_id": post_id,
            "post_text": text,
            "true_label": true_label,
            "predicted": pred_label,
            "correct": is_correct
        })

#计算大模型准确率
    total = len(results)
    correct = sum(1 for r in results if r['correct'])

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

# 将运行结果保存为result_no_sentiment.csv
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "post_id", "post_text", "true_label", "predicted", "correct"
        ])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    dataset = load_dataset(INPUT_FILE, limit=LIMIT)
    run_analysis(dataset)
