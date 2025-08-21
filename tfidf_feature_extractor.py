"""
改进的TF-IDF特征提取器
根据词表赋权重，适用于中文谣言检测
"""

import numpy as np
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re
import pickle
import os

class ImprovedTFIDFExtractor:
    """
    改进的TF-IDF特征提取器
    - 使用jieba分词
    - 支持自定义词表权重
    - 过滤停用词
    - 支持谣言相关关键词加权
    """
    
    def __init__(self, max_features=10000, min_df=2, max_df=0.95):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None
        self.word_weights = {}
        self.stop_words = set()
        self.rumor_keywords = set()
        
        # 初始化jieba
        jieba.initialize()
        
        # 加载停用词
        self._load_stop_words()
        
        # 定义谣言相关关键词（可以根据需要扩展）
        self._init_rumor_keywords()
    
    def _load_stop_words(self):
        """加载中文停用词表"""
        # 基础中文停用词
        basic_stop_words = [
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '之后', '时候', '可以', '还', '把', '比', '或者', '已经',
            '但是', '如果', '只是', '当', '而且', '对于', '因为', '所以', '虽然', '然后',
            '但', '与', '及', '以及', '等等', '什么', '怎么', '为什么', '哪里', '哪个',
            '多少', '几个', '这样', '那样', '这里', '那里', '现在', '以前', '之前',
            '以后', '将来', '今天', '昨天', '明天', '年', '月', '日', '点', '分', '秒'
        ]
        self.stop_words.update(basic_stop_words)
    
    def _init_rumor_keywords(self):
        """初始化谣言相关关键词"""
        rumor_keywords = [
            '紧急', '扩散', '转发', '求转', '速转', '快转', '告诉', '通知', '警告', '注意',
            '谣言', '假消息', '不实', '虚假', '传言', '听说', '据说', '爆料', '内幕',
            '震惊', '惊人', '恐怖', '可怕', '危险', '严重', '紧急情况', '突发',
            '官方', '权威', '专家', '医生', '教授', '科学家', '研究表明', '证实',
            '死亡', '致癌', '有毒', '污染', '感染', '病毒', '细菌', '疾病',
            '地震', '火灾', '爆炸', '事故', '灾难', '洪水', '台风',
            '涨价', '降价', '免费', '优惠', '打折', '促销', '限时',
            '政策', '法律', '规定', '禁止', '允许', '批准', '取消'
        ]
        self.rumor_keywords.update(rumor_keywords)
    
    def set_word_weights(self, word_weight_dict):
        """
        设置词汇权重
        Args:
            word_weight_dict: 词汇->权重的字典
        """
        self.word_weights.update(word_weight_dict)
    
    def _preprocess_text(self, text):
        """
        文本预处理
        Args:
            text: 原始文本
        Returns:
            处理后的文本
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 清理文本
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'@[^\s]+', '', text)  # 移除@用户名
        text = re.sub(r'#[^#]+#', '', text)  # 移除话题标签
        text = re.sub(r'[^\u4e00-\u9fff\w\s]', '', text)  # 只保留中文、英文、数字
        
        # 使用jieba分词
        words = jieba.cut(text)
        
        # 过滤停用词和短词
        filtered_words = []
        for word in words:
            word = word.strip()
            if len(word) > 1 and word not in self.stop_words:
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def _custom_tokenizer(self, text):
        """自定义分词器"""
        return self._preprocess_text(text).split()
    
    def fit(self, texts):
        """
        训练TF-IDF向量化器
        Args:
            texts: 文本列表
        """
        # 预处理文本
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # 创建TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            tokenizer=lambda x: x.split(),  # 已经预处理过了
            lowercase=False,
            ngram_range=(1, 2)  # 使用1-gram和2-gram
        )
        
        # 训练向量化器
        self.vectorizer.fit(processed_texts)
        
        # 分析词汇重要性，为谣言关键词设置更高权重
        feature_names = self.vectorizer.get_feature_names_out()
        for word in feature_names:
            if word in self.rumor_keywords:
                self.word_weights[word] = 2.0  # 谣言关键词权重加倍
            elif any(keyword in word for keyword in self.rumor_keywords):
                self.word_weights[word] = 1.5  # 包含谣言关键词的词组权重增加
    
    def transform(self, texts):
        """
        将文本转换为TF-IDF特征向量
        Args:
            texts: 文本列表
        Returns:
            特征矩阵 [N, feature_dim]
        """
        if self.vectorizer is None:
            raise ValueError("必须先调用fit方法训练向量化器")
        
        # 预处理文本
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # 获取TF-IDF矩阵
        tfidf_matrix = self.vectorizer.transform(processed_texts)
        
        # 应用自定义权重
        if self.word_weights:
            feature_names = self.vectorizer.get_feature_names_out()
            for i, word in enumerate(feature_names):
                if word in self.word_weights:
                    tfidf_matrix[:, i] *= self.word_weights[word]
        
        return tfidf_matrix.toarray()
    
    def fit_transform(self, texts):
        """训练并转换文本"""
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self):
        """获取特征名称"""
        if self.vectorizer is None:
            return []
        return self.vectorizer.get_feature_names_out()
    
    def save_model(self, filepath):
        """保存模型"""
        model_data = {
            'vectorizer': self.vectorizer,
            'word_weights': self.word_weights,
            'stop_words': self.stop_words,
            'rumor_keywords': self.rumor_keywords
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.word_weights = model_data['word_weights']
        self.stop_words = model_data['stop_words']
        self.rumor_keywords = model_data['rumor_keywords']

def extract_tfidf_features(texts, max_features=10000, custom_weights=None):
    """
    便捷函数：提取改进的TF-IDF特征
    
    Args:
        texts: 文本列表
        max_features: 最大特征数
        custom_weights: 自定义词汇权重字典
    
    Returns:
        features: TF-IDF特征矩阵 [N, feature_dim]
        extractor: 训练好的特征提取器
    """
    extractor = ImprovedTFIDFExtractor(max_features=max_features)
    
    if custom_weights:
        extractor.set_word_weights(custom_weights)
    
    features = extractor.fit_transform(texts)
    
    print(f"TF-IDF特征提取完成:")
    print(f"- 文本数量: {len(texts)}")
    print(f"- 特征维度: {features.shape[1]}")
    print(f"- 特征矩阵形状: {features.shape}")
    
    return features, extractor

# 使用示例
if __name__ == "__main__":
    # 示例文本
    sample_texts = [
        "紧急扩散！这个消息一定要转发给所有人看到！",
        "今天天气很好，适合出门散步。",
        "据专家研究表明，这种食品含有致癌物质，大家千万不要吃！",
        "刚刚看到一个有趣的新闻，分享给大家。"
    ]
    
    # 提取TF-IDF特征
    features, extractor = extract_tfidf_features(sample_texts, max_features=1000)
    
    print("\n特征矩阵示例:")
    print(features[:2])  # 打印前两个样本的特征