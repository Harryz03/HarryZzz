import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import pyLDAvis.gensim_models
import pyLDAvis
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import urllib.request
import json

# 1. 数据准备
# 只需 post_text 字段，其他字段已删除
data = pd.read_csv('DATA2.txt', sep='\t')
texts = data['post_text'].tolist()

# 2. 数据预处理
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # 去除非字母字符，转小写
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    # 分词
    tokens = text.split()
    # 去停用词和短词
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    # 词形还原
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return tokens

processed_texts = [preprocess(t) for t in texts]

# 3. 构建词典和语料库
dictionary = corpora.Dictionary(processed_texts)
corpus = [dictionary.doc2bow(text) for text in processed_texts]

# 4. 训练LDA模型
lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20, random_state=42)

# 5. 可视化分析
# 5.1 pyLDAvis
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis, 'lda_vis.html')
print("pyLDAvis结果已保存为lda_vis.html")

# 5.2 词云图
for t in range(lda_model.num_topics):
    plt.figure()
    plt.title(f"Topic #{t+1} WordCloud")
    topic_words = dict(lda_model.show_topic(t, 20))
    wc = WordCloud(width=600, height=400, background_color='white').generate_from_frequencies(topic_words)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(f'topic{t+1}_wordcloud.png')
    plt.close()
print("词云图已保存为topic*_wordcloud.png")

# 5.3 (可选) 热力图
topic_matrix = np.zeros((len(corpus), lda_model.num_topics))
for i, bow in enumerate(corpus):
    for tid, prob in lda_model.get_document_topics(bow):
        topic_matrix[i, tid] = prob
plt.figure(figsize=(8, 4))
plt.imshow(topic_matrix, aspect='auto', cmap='YlGnBu', alpha=0.70)  # 增加透明度
plt.colorbar(label='Topic Probability')
plt.xlabel('Topic')
plt.ylabel('Document')
plt.title('Document-Topic Heatmap')
plt.savefig('doc_topic_heatmap.png')
plt.close()
print("热力图已保存为doc_topic_heatmap.png")

# 6. 结合大模型分析主题内容（示例：用GPT风格总结）
def summarize_topic(topic_words):
    # 这里只做简单总结，实际可用大模型API
    summary = "该主题关键词包括：" + ", ".join([w for w, _ in topic_words])
    return summary

print("\n主题内容总结：")
for t in range(lda_model.num_topics):
    topic_words = lda_model.show_topic(t, 10)
    print(f"主题{t+1}：{summarize_topic(topic_words)}")

# 6. 结合大模型分析主题内容（通过Ollama本地Gemma3:12b-it-qat自动总结）
def summarize_topic_with_ollama(topic_words):
    prompt = (
        "请用简洁的语言总结以下关键词代表的主题内容：\n" +
        ", ".join([w for w, _ in topic_words])
    )
    data = json.dumps({
        "model": "gemma3:12b-it-qat",
        "prompt": prompt,
        "stream": False
    }).encode('utf-8')
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=data,
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req) as f:
            resp = f.read().decode('utf-8')
            result = json.loads(resp)
            return result.get("response", "").strip()
    except Exception as e:
        return f"Ollama请求失败: {e}"

print("\n主题内容总结（Gemma3大模型）：")
for t in range(lda_model.num_topics):
    topic_words = lda_model.show_topic(t, 10)
    summary = summarize_topic_with_ollama(topic_words)
    print(f"主题{t+1}：{summary}")
