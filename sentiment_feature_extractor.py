"""
情感分析分数特征提取器
基于词典和机器学习方法进行中文情感分析
"""

import numpy as np
import jieba
import jieba.analyse
from collections import Counter
import re
import os

class SentimentFeatureExtractor:
    """
    情感分析特征提取器
    提取多维度情感特征，包括正负情感分数、情感强度、情感词频等
    """
    
    def __init__(self):
        self.positive_words = set()
        self.negative_words = set()
        self.degree_words = {}  # 程度副词及其权重
        self.negation_words = set()  # 否定词
        
        # 初始化jieba
        jieba.initialize()
        
        # 加载情感词典
        self._load_sentiment_lexicon()
        
        # 加载程度副词
        self._load_degree_words()
        
        # 加载否定词
        self._load_negation_words()
    
    def _load_sentiment_lexicon(self):
        """加载情感词典"""
        # 正面情感词
        positive_words = [
            '好', '棒', '优秀', '完美', '杰出', '卓越', '精彩', '美好', '幸福', '快乐',
            '开心', '高兴', '兴奋', '满意', '赞', '支持', '认同', '同意', '正确', '对',
            '真实', '可靠', '值得', '信任', '安全', '健康', '有效', '成功', '胜利',
            '希望', '乐观', '积极', '正面', '光明', '美丽', '漂亮', '可爱', '温暖',
            '感谢', '谢谢', '感激', '感动', '温馨', '舒适', '放心', '踏实', '稳定'
        ]
        
        # 负面情感词
        negative_words = [
            '坏', '差', '糟糕', '恶劣', '可怕', '恐怖', '危险', '有害', '毒', '假',
            '虚假', '不实', '谣言', '骗子', '欺骗', '诈骗', '陷阱', '威胁', '攻击',
            '伤害', '损害', '破坏', '污染', '腐败', '黑暗', '邪恶', '恶心', '讨厌',
            '愤怒', '生气', '愤慨', '不满', '抱怨', '批评', '指责', '谴责', '反对',
            '拒绝', '否定', '错误', '失败', '失望', '绝望', '悲伤', '痛苦', '难过',
            '担心', '焦虑', '紧张', '害怕', '恐惧', '慌张', '混乱', '麻烦', '问题'
        ]
        
        self.positive_words.update(positive_words)
        self.negative_words.update(negative_words)
    
    def _load_degree_words(self):
        """加载程度副词及其权重"""
        # 程度副词：词汇 -> 权重
        degree_words = {
            # 强化程度
            '非常': 2.0, '特别': 2.0, '十分': 2.0, '极其': 2.5, '极度': 2.5,
            '超级': 2.0, '超': 1.8, '巨': 1.8, '超级': 2.0, '绝对': 2.2,
            '完全': 2.0, '彻底': 2.0, '深深': 1.8, '强烈': 1.8, '严重': 1.8,
            
            # 中等程度
            '很': 1.5, '比较': 1.3, '相当': 1.4, '挺': 1.3, '蛮': 1.3,
            '还': 1.2, '颇': 1.3, '相对': 1.2, '稍微': 1.1, '略': 1.1,
            
            # 轻微程度
            '有点': 1.1, '一点': 1.1, '稍': 1.1, '些许': 1.1, '略微': 1.1,
            '一些': 1.1, '某种': 1.1, '多少': 1.1, '几乎': 0.9, '差不多': 0.9
        }
        
        self.degree_words.update(degree_words)
    
    def _load_negation_words(self):
        """加载否定词"""
        negation_words = [
            '不', '没', '无', '非', '未', '否', '别', '莫', '勿', '毋',
            '不是', '不会', '不能', '不要', '不用', '没有', '没法', '没办法',
            '并非', '并不', '绝非', '决不', '从不', '永不', '从未', '未曾',
            '难以', '无法', '不可', '不得', '禁止', '拒绝', '反对'
        ]
        
        self.negation_words.update(negation_words)
    
    def _analyze_sentence_sentiment(self, words):
        """
        分析句子情感
        
        Args:
            words: 分词后的词列表
        
        Returns:
            sentiment_score: 情感分数
            sentiment_features: 详细情感特征
        """
        positive_score = 0
        negative_score = 0
        degree_modifier = 1.0
        negation_count = 0
        
        for i, word in enumerate(words):
            # 检查程度副词
            if word in self.degree_words:
                degree_modifier = self.degree_words[word]
                continue
            
            # 检查否定词
            if word in self.negation_words:
                negation_count += 1
                continue
            
            # 检查情感词
            if word in self.positive_words:
                score = 1.0 * degree_modifier
                if negation_count % 2 == 1:  # 奇数个否定词表示否定
                    negative_score += score
                else:
                    positive_score += score
            elif word in self.negative_words:
                score = 1.0 * degree_modifier
                if negation_count % 2 == 1:  # 奇数个否定词表示否定
                    positive_score += score
                else:
                    negative_score += score
            
            # 重置程度修饰符
            degree_modifier = 1.0
        
        # 计算综合情感分数
        total_score = positive_score - negative_score
        
        # 归一化
        max_possible = len(words) * 2.5  # 假设最大程度副词权重为2.5
        if max_possible > 0:
            normalized_score = total_score / max_possible
        else:
            normalized_score = 0
        
        # 情感特征
        sentiment_features = {
            'positive_score': positive_score,
            'negative_score': negative_score,
            'total_score': total_score,
            'normalized_score': normalized_score,
            'sentiment_intensity': abs(normalized_score),
            'positive_word_count': sum(1 for w in words if w in self.positive_words),
            'negative_word_count': sum(1 for w in words if w in self.negative_words),
            'degree_word_count': sum(1 for w in words if w in self.degree_words),
            'negation_count': sum(1 for w in words if w in self.negation_words)
        }
        
        return normalized_score, sentiment_features
    
    def _extract_rumor_sentiment_features(self, words):
        """提取谣言特定的情感特征"""
        # 谣言相关的情感词汇
        rumor_positive_words = {'真实', '可信', '证实', '确认', '官方', '权威', '专业'}
        rumor_negative_words = {'假', '虚假', '谣言', '不实', '欺骗', '误导', '夸大', '炒作'}
        
        rumor_pos_count = sum(1 for w in words if w in rumor_positive_words)
        rumor_neg_count = sum(1 for w in words if w in rumor_negative_words)
        
        # 紧急性词汇
        urgency_words = {'紧急', '立即', '马上', '赶紧', '快', '速度', '急'}
        urgency_count = sum(1 for w in words if w in urgency_words)
        
        # 传播性词汇
        spread_words = {'转发', '扩散', '传播', '告诉', '通知', '分享', '转'}
        spread_count = sum(1 for w in words if w in spread_words)
        
        return {
            'rumor_positive_count': rumor_pos_count,
            'rumor_negative_count': rumor_neg_count,
            'urgency_score': urgency_count / len(words) if words else 0,
            'spread_score': spread_count / len(words) if words else 0
        }
    
    def extract_sentiment_features(self, text):
        """
        提取单个文本的情感特征
        
        Args:
            text: 输入文本
        
        Returns:
            feature_vector: 情感特征向量
        """
        if not text or not isinstance(text, str):
            return np.zeros(13)  # 返回13维零向量
        
        # 预处理和分词
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'@[^\s]+', '', text)
        text = re.sub(r'#[^#]+#', '', text)
        
        words = list(jieba.cut(text))
        words = [word.strip() for word in words if len(word.strip()) > 0]
        
        # 基础情感分析
        sentiment_score, sentiment_features = self._analyze_sentence_sentiment(words)
        
        # 谣言特定情感特征
        rumor_features = self._extract_rumor_sentiment_features(words)
        
        # 构建特征向量
        feature_vector = np.array([
            sentiment_features['positive_score'],
            sentiment_features['negative_score'],
            sentiment_features['total_score'],
            sentiment_features['normalized_score'],
            sentiment_features['sentiment_intensity'],
            sentiment_features['positive_word_count'],
            sentiment_features['negative_word_count'],
            sentiment_features['degree_word_count'],
            sentiment_features['negation_count'],
            rumor_features['rumor_positive_count'],
            rumor_features['rumor_negative_count'],
            rumor_features['urgency_score'],
            rumor_features['spread_score']
        ])
        
        return feature_vector
    
    def extract_features(self, texts):
        """
        批量提取情感特征
        
        Args:
            texts: 文本列表
        
        Returns:
            features: 情感特征矩阵 [N, 13]
        """
        print(f"正在提取 {len(texts)} 个文本的情感分析特征...")
        
        features = []
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                print(f"进度: {i}/{len(texts)}")
            
            feature_vector = self.extract_sentiment_features(text)
            features.append(feature_vector)
        
        features = np.array(features)
        print(f"情感分析特征提取完成，特征矩阵形状: {features.shape}")
        
        return features
    
    def get_feature_names(self):
        """获取特征名称"""
        return [
            'positive_score',      # 正面情感分数
            'negative_score',      # 负面情感分数
            'total_score',         # 总情感分数
            'normalized_score',    # 归一化情感分数
            'sentiment_intensity', # 情感强度
            'positive_word_count', # 正面词数量
            'negative_word_count', # 负面词数量
            'degree_word_count',   # 程度副词数量
            'negation_count',      # 否定词数量
            'rumor_positive_count',# 谣言正面词数量
            'rumor_negative_count',# 谣言负面词数量
            'urgency_score',       # 紧急性分数
            'spread_score'         # 传播性分数
        ]

def extract_sentiment_features(texts):
    """
    便捷函数：提取情感分析特征
    
    Args:
        texts: 文本列表
    
    Returns:
        features: 情感特征矩阵 [N, 13]
        extractor: 情感分析提取器
        feature_names: 特征名称列表
    """
    extractor = SentimentFeatureExtractor()
    features = extractor.extract_features(texts)
    feature_names = extractor.get_feature_names()
    
    print(f"情感分析特征提取完成:")
    print(f"- 文本数量: {len(texts)}")
    print(f"- 特征维度: {features.shape[1]}")
    print(f"- 特征名称: {feature_names}")
    
    return features, extractor, feature_names

# 高级情感分析器（基于预训练模型）
class AdvancedSentimentAnalyzer:
    """
    高级情感分析器
    使用预训练的情感分析模型（如果可用）
    """
    
    def __init__(self, model_name='bert-base-chinese'):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._init_model()
    
    def _init_model(self):
        """初始化预训练情感分析模型"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            # 尝试加载中文情感分析模型
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
            print(f"成功加载预训练情感分析模型: {self.model_name}")
        except Exception as e:
            print(f"预训练模型加载失败: {e}")
            print("将使用基础词典方法")
            self.model = None
    
    def predict_sentiment(self, text):
        """
        预测文本情感
        
        Args:
            text: 输入文本
        
        Returns:
            sentiment_scores: [negative_prob, positive_prob]
        """
        if self.model is None:
            # 回退到基础方法
            extractor = SentimentFeatureExtractor()
            features = extractor.extract_sentiment_features(text)
            # 简单映射到概率
            normalized_score = features[3]  # normalized_score
            if normalized_score > 0:
                return [0.3, 0.7]  # 正面
            elif normalized_score < 0:
                return [0.7, 0.3]  # 负面
            else:
                return [0.5, 0.5]  # 中性
        
        try:
            import torch
            
            # 编码文本
            inputs = self.tokenizer(text, return_tensors='pt', 
                                  truncation=True, max_length=512, 
                                  padding=True)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                return probabilities.squeeze().numpy()
        
        except Exception as e:
            print(f"情感预测失败: {e}")
            return [0.5, 0.5]  # 返回中性

# 使用示例
if __name__ == "__main__":
    # 示例文本
    sample_texts = [
        "紧急扩散！这个消息一定要转发给所有人看到！非常危险！",
        "今天天气很好，心情也很棒，感谢大家的支持。",
        "据专家研究表明，这种食品含有致癌物质，大家千万不要吃！",
        "刚刚看到一个有趣的新闻，分享给大家，希望大家喜欢。"
    ]
    
    # 基础情感分析
    print("=== 基础情感分析特征 ===")
    features, extractor, feature_names = extract_sentiment_features(sample_texts)
    
    print("\n特征矩阵示例:")
    for i, text in enumerate(sample_texts):
        print(f"\n文本 {i+1}: {text[:30]}...")
        print(f"情感特征: {features[i]}")
        print(f"主要情感分数: {features[i][3]:.3f}")
    
    # 高级情感分析
    print("\n=== 高级情感分析 ===")
    try:
        advanced_analyzer = AdvancedSentimentAnalyzer()
        for i, text in enumerate(sample_texts):
            sentiment_probs = advanced_analyzer.predict_sentiment(text)
            print(f"文本 {i+1} 情感概率: 负面={sentiment_probs[0]:.3f}, 正面={sentiment_probs[1]:.3f}")
    except Exception as e:
        print(f"高级情感分析失败: {e}")
        print("请安装transformers库: pip install transformers torch")