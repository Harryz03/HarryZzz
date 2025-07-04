# HarryZzz
云计算与大数据分析---新闻谣言检测系统
题目：多源融合新闻真伪判别模型
Cloud
├── Cloud1   ### 一、基于大模型的情感语义分析
│   ├── fake_news_1.py   ### 新闻文本判别
│   ├── new_sensetive_1.py   ### 新闻文本情感分析
│   ├── news_fakeness_sensetive.py   ### 情感辅助下的新闻文本真假判别
│   └── posts_groundtruth.txt   ### 数据集（使用前1000条）
├── Cloud2   ### 二、基于大模型的twitter主题分析
│   ├── DATA2.txt   ### 数据集
│   ├── doc_topic_heatmap.png   ### 热力图
│   ├── lda_vis.html   ### 可视化网页
│   ├── topic1_wordcloud.png   ### 主题词组1
│   ├── topic2_wordcloud.png   ### 主题词组2
│   ├── topic3_wordcloud.png   ### 主题词组3
│   └── twitter_ana.py   ### 基于大模型的twitter主题分析代码
├── Cloud3   ### 三、多模态（情感、主题语义）综合预测与分析
│   ├── __pycache__
│   ├── images   ### 数据集（图像）
│   ├── entity_knowledge.json   ### 数据集（Wikipedia摘要）
│   ├── eric_fnd_model.py   ### 模型代码
│   ├── news_dataset.json   ### 数据集（文本）
│   └── run_model.py   ### 模型运行代码
└── README.md
