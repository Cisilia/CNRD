# 谣言检测特征提取系统 - 快速开始

## 🎯 系统概述

基于您的中文谣言数据集，我已经为您构建了完整的三种文本特征提取系统：

1. **改进的TF-IDF特征** - 基于词频-逆文档频率，支持谣言关键词加权
2. **词嵌入特征** - 支持Word2Vec、BERT等多种词向量方法
3. **情感分析特征** - 13维情感指标，包含谣言特定特征

## 📁 文件结构

```
/workspace/
├── 数据文件
│   ├── rumors_v170613.json          # 主要谣言数据集 (31,669条)
│   └── CED_Dataset/                 # 包含转发评论的数据集
│
├── 特征提取器
│   ├── tfidf_feature_extractor.py   # TF-IDF特征提取器
│   ├── word_embedding_extractor.py  # 词嵌入特征提取器
│   └── sentiment_feature_extractor.py # 情感分析特征提取器
│
├── 核心模块
│   ├── data_loader.py               # 数据加载和预处理
│   ├── rumor_detection_features.py  # 主程序（整合所有特征）
│   └── usage_example.py             # 使用示例
│
├── 演示和文档
│   ├── simple_feature_demo.py       # 简化版演示（无外部依赖）
│   ├── FEATURE_EXTRACTION_README.md # 详细说明文档
│   ├── requirements.txt             # 依赖包列表
│   └── demo_features.txt            # 演示结果（已生成）
```

## 🚀 快速运行

### 方法1：简化版演示（推荐开始）
```bash
# 无需安装任何依赖，直接运行
python3 simple_feature_demo.py
```

### 方法2：完整版本（需要安装依赖）
```bash
# 安装依赖
pip install -r requirements.txt

# 运行完整演示
python3 usage_example.py

# 运行完整训练
python3 rumor_detection_features.py
```

## 📊 特征说明

### 1. TF-IDF特征 (可配置维度)
- **原理**: 词频-逆文档频率，反映词汇重要性
- **改进**: 谣言关键词权重加倍，停用词过滤
- **输出**: 稀疏向量，维度可调 (默认5000维)

### 2. 词嵌入特征 (默认300维)
- **方法**: Word2Vec/BERT/GloVe
- **原理**: 将文本映射到连续向量空间
- **输出**: 密集向量，捕获语义信息

### 3. 情感分析特征 (固定13维)
```
[正面分数, 负面分数, 总分数, 归一化分数, 情感强度, 
 正面词数, 负面词数, 程度词数, 否定词数, 
 谣言正面词数, 谣言负面词数, 紧急性分数, 传播性分数]
```

## 💡 使用示例

### 单独使用各特征提取器

```python
# 1. TF-IDF特征
from tfidf_feature_extractor import extract_tfidf_features

texts = ["您的文本列表"]
tfidf_features, extractor = extract_tfidf_features(texts, max_features=5000)
print(f"TF-IDF特征形状: {tfidf_features.shape}")

# 2. 词嵌入特征  
from word_embedding_extractor import extract_word_embedding_features

embedding_features, extractor = extract_word_embedding_features(
    texts, 
    method='word2vec',  # 或 'bert'
    embedding_dim=300,
    train_new_model=True
)
print(f"词嵌入特征形状: {embedding_features.shape}")

# 3. 情感分析特征
from sentiment_feature_extractor import extract_sentiment_features

sentiment_features, extractor, feature_names = extract_sentiment_features(texts)
print(f"情感特征形状: {sentiment_features.shape}")
print(f"特征名称: {feature_names}")
```

### 整合使用

```python
from rumor_detection_features import RumorDetectionFeatureExtractor

# 初始化特征提取器
extractor = RumorDetectionFeatureExtractor(
    tfidf_max_features=3000,
    embedding_dim=200,
    embedding_method='word2vec'
)

# 提取所有特征
combined_features, feature_info = extractor.extract_all_features(texts)
print(f"合并特征形状: {combined_features.shape}")
```

## 🔧 自定义配置

### TF-IDF自定义权重
```python
# 设置特定词汇的权重
custom_weights = {
    '紧急': 3.0,    # 紧急相关词汇权重增加
    '扩散': 2.5,    # 传播相关词汇
    '专家': 2.0,    # 权威性词汇
    '据说': 1.5     # 不确定性词汇
}

tfidf_features, extractor = extract_tfidf_features(
    texts, 
    max_features=10000,
    custom_weights=custom_weights
)
```

## 📈 演示结果

运行 `simple_feature_demo.py` 后，您会看到：

- ✅ 成功加载了50条谣言数据 + 10条非谣言数据
- ✅ TF-IDF特征: 315维向量
- ✅ 词嵌入特征: 50维向量  
- ✅ 情感分析特征: 9维向量
- ✅ 合并特征: 374维向量
- ✅ 详细结果保存在 `demo_features.txt`

## 🎯 下一步

1. **安装完整依赖**: 运行 `pip install -r requirements.txt`
2. **使用更多数据**: 修改 `max_samples` 参数加载更多数据
3. **调整特征维度**: 根据需要调整各特征的维度
4. **训练分类器**: 使用提取的特征训练机器学习模型
5. **模型评估**: 使用交叉验证评估模型性能

## ⚡ 性能提示

- 小数据集（<1000样本）: 使用 `simple_feature_demo.py`
- 中等数据集（1000-10000样本）: 使用 `usage_example.py`  
- 大数据集（>10000样本）: 使用 `rumor_detection_features.py` 并分批处理

## 🛠️ 故障排除

**问题**: 缺少依赖包
**解决**: 先运行简化版 `simple_feature_demo.py`，再安装依赖

**问题**: 内存不足
**解决**: 减少 `max_samples` 参数，分批处理数据

**问题**: 中文分词效果不好
**解决**: 安装jieba后使用完整版本

---

🎉 **恭喜！您的谣言检测特征提取系统已经准备就绪！**