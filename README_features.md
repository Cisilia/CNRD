## 特征提取脚本使用说明

依赖安装：

```bash
pip install -r /workspace/requirements.txt
```

数据目录：`/workspace/CED_Dataset`

### 1) 改进的 TF-IDF（按词表赋权重）

可选传入一个 JSON 词权重表：`{"词": 权重}`，将对对应词的 TF-IDF 列乘以该权重。

```bash
python /workspace/src/features_tfidf.py \
  --ced_dir /workspace/CED_Dataset \
  --out_dir /workspace/outputs/tfidf \
  --max_features 50000 \
  --min_df 2 \
  --max_df 0.95 \
  --ngram_max 2 \
  --vocab_weights /workspace/vocab_weights.json   # 可选
```

输出：
- `/workspace/outputs/tfidf/tfidf_features.npz` 稀疏矩阵 (N, D)
- `/workspace/outputs/tfidf/feature_names.json`
- `/workspace/outputs/tfidf/labels.json`（含 `ids` 与 `labels`）
- `/workspace/outputs/tfidf/tfidf_vectorizer.joblib`

### 2) 句向量/词嵌入特征（Sentence-Transformers）

默认使用多语种中文模型 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`。

```bash
python /workspace/src/features_embeddings.py \
  --ced_dir /workspace/CED_Dataset \
  --out_dir /workspace/outputs/embeddings \
  --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

输出：
- `/workspace/outputs/embeddings/embeddings.npy` (N, dim)
- `/workspace/outputs/embeddings/labels.json`

### 3) 情感分析分数特征

使用中文二分类情感模型 `uer/roberta-base-finetuned-jd-binary-chinese`，输出每条文本的 `[neg_prob, pos_prob]`。

```bash
python /workspace/src/features_sentiment.py \
  --ced_dir /workspace/CED_Dataset \
  --out_dir /workspace/outputs/sentiment
```

输出：
- `/workspace/outputs/sentiment/sentiment.npy` (N, 2)
- `/workspace/outputs/sentiment/labels.json`

说明：
- `ced_loader.py` 将 `original-microblog` 与 `rumor-repost`/`non-rumor-repost` 通过文件名一致性进行匹配，生成标签：`1=谣言`，`0=非谣言`。
- 三个脚本输出的样本顺序一致，可用 `labels.json` 中的 `ids` 对齐。

