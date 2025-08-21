# 谣言检测特征提取系统

本系统基于您上传的中文谣言数据集，实现了三种文本特征提取方法，用于谣言检测任务。

## 特征类型

### 1. 改进的TF-IDF特征 (`tfidf_feature_extractor.py`)
- 基于jieba中文分词
- 支持自定义词汇权重
- 谣言关键词加权处理
- 停用词过滤
- 支持1-gram和2-gram特征

### 2. 词嵌入特征 (`word_embedding_extractor.py`)
- 支持多种词嵌入方法：
  - Word2Vec（可训练新模型）
  - BERT中文预训练模型
  - GloVe（需要预训练模型）
  - 平均词向量
- 自动文本预处理和分词
- 灵活的嵌入维度设置

### 3. 情感分析特征 (`sentiment_feature_extractor.py`)
- 基于中文情感词典
- 13维情感特征：
  - 正负情感分数
  - 情感强度
  - 情感词频统计
  - 程度副词和否定词分析
  - 谣言特定情感指标（紧急性、传播性等）

## 文件说明

- `tfidf_feature_extractor.py` - TF-IDF特征提取器
- `word_embedding_extractor.py` - 词嵌入特征提取器  
- `sentiment_feature_extractor.py` - 情感分析特征提取器
- `data_loader.py` - 数据加载和预处理模块
- `rumor_detection_features.py` - 主程序，整合所有特征提取器
- `usage_example.py` - 使用示例和演示
- `requirements.txt` - 依赖包列表

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行演示

```bash
# 运行特征提取演示
python usage_example.py

# 运行完整的模型训练
python rumor_detection_features.py

# 仅提取特征（不训练模型）
python rumor_detection_features.py --extract-only
```

### 3. 使用单个特征提取器

```python
# TF-IDF特征
from tfidf_feature_extractor import extract_tfidf_features
features, extractor = extract_tfidf_features(texts, max_features=5000)

# 词嵌入特征
from word_embedding_extractor import extract_word_embedding_features
embeddings, extractor = extract_word_embedding_features(
    texts, method='word2vec', embedding_dim=300, train_new_model=True
)

# 情感分析特征
from sentiment_feature_extractor import extract_sentiment_features
sentiment_features, extractor, feature_names = extract_sentiment_features(texts)
```

## 数据集支持

系统支持两种数据集格式：

1. **基础谣言数据集** (`rumors_v170613.json`)
   - 31,669条谣言数据
   - JSON格式，每行一条记录
   - 包含谣言文本、标题、审查结果等

2. **CED数据集** (`CED_Dataset/`)
   - 包含谣言和非谣言样本
   - 1,538条谣言 + 1,849条非谣言
   - 包含转发和评论信息

## 特征输出格式

所有特征提取器都返回numpy数组格式：

- **TF-IDF特征**: `[N, tfidf_dim]` - 稀疏文本特征
- **词嵌入特征**: `[N, embedding_dim]` - 密集语义特征  
- **情感分析特征**: `[N, 13]` - 多维情感指标

其中N是样本数量。

## 自定义配置

### TF-IDF配置
```python
# 自定义词汇权重
custom_weights = {
    '紧急': 2.0,
    '扩散': 1.5,
    '专家': 1.8
}

features, extractor = extract_tfidf_features(
    texts, 
    max_features=10000,
    custom_weights=custom_weights
)
```

### 词嵌入配置
```python
# 使用BERT
embeddings, extractor = extract_word_embedding_features(
    texts, 
    method='bert',
    model_path='bert-base-chinese'
)

# 训练Word2Vec
embeddings, extractor = extract_word_embedding_features(
    texts,
    method='word2vec', 
    embedding_dim=300,
    train_new_model=True
)
```

## 注意事项

1. **内存使用**: 大数据集可能需要大量内存，建议分批处理
2. **计算时间**: BERT特征提取较慢，Word2Vec训练需要时间
3. **模型保存**: 所有训练的模型都会自动保存到`models/`目录
4. **特征保存**: 提取的特征可保存为CSV和NPY格式

## 扩展功能

- 支持添加自定义情感词典
- 支持多种预训练词向量
- 支持特征选择和降维
- 支持增量学习和模型更新

## 性能优化建议

1. 对于大数据集，使用`max_samples`参数限制样本数
2. TF-IDF特征数量不宜过大（建议3000-10000）
3. 词嵌入维度建议100-300维
4. 可以使用GPU加速BERT特征提取