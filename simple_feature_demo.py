"""
简化版谣言检测特征提取演示
使用Python标准库实现核心功能，无需外部依赖
"""

import json
import re
import math
from collections import Counter, defaultdict
import random

class SimpleTFIDFExtractor:
    """简化的TF-IDF特征提取器"""
    
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vocabulary = {}
        self.idf_scores = {}
        self.rumor_keywords = {
            '紧急', '扩散', '转发', '求转', '速转', '快转', '告诉', '通知', '警告', '注意',
            '谣言', '假消息', '不实', '虚假', '传言', '听说', '据说', '爆料', '内幕',
            '震惊', '惊人', '恐怖', '可怕', '危险', '严重', '紧急情况', '突发',
            '官方', '权威', '专家', '医生', '教授', '科学家', '研究表明', '证实'
        }
    
    def simple_tokenize(self, text):
        """简单的中文分词（基于字符和标点）"""
        if not text:
            return []
        
        # 清理文本
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'@\S+', '', text)
        text = re.sub(r'#[^#]+#', '', text)
        
        # 简单分词：按空格、标点分割，保留中文词汇
        words = re.findall(r'[\u4e00-\u9fff]+', text)
        
        # 过滤短词
        words = [word for word in words if len(word) >= 2]
        
        return words
    
    def build_vocabulary(self, texts):
        """构建词汇表"""
        word_freq = Counter()
        
        for text in texts:
            words = self.simple_tokenize(text)
            word_freq.update(words)
        
        # 选择最频繁的词汇
        most_common = word_freq.most_common(self.max_features)
        self.vocabulary = {word: idx for idx, (word, freq) in enumerate(most_common)}
        
        print(f"词汇表大小: {len(self.vocabulary)}")
        return self.vocabulary
    
    def compute_idf(self, texts):
        """计算IDF分数"""
        doc_count = len(texts)
        word_doc_count = defaultdict(int)
        
        for text in texts:
            words = set(self.simple_tokenize(text))
            for word in words:
                if word in self.vocabulary:
                    word_doc_count[word] += 1
        
        for word in self.vocabulary:
            df = word_doc_count[word]
            idf = math.log(doc_count / (df + 1))
            
            # 谣言关键词加权
            if word in self.rumor_keywords:
                idf *= 2.0
            
            self.idf_scores[word] = idf
    
    def extract_features(self, texts):
        """提取TF-IDF特征"""
        if not self.vocabulary:
            self.build_vocabulary(texts)
            self.compute_idf(texts)
        
        features = []
        
        for text in texts:
            words = self.simple_tokenize(text)
            word_count = Counter(words)
            doc_length = len(words)
            
            # 计算TF-IDF向量
            tfidf_vector = [0.0] * len(self.vocabulary)
            
            for word, count in word_count.items():
                if word in self.vocabulary:
                    tf = count / doc_length if doc_length > 0 else 0
                    idf = self.idf_scores.get(word, 0)
                    tfidf = tf * idf
                    
                    word_idx = self.vocabulary[word]
                    tfidf_vector[word_idx] = tfidf
            
            features.append(tfidf_vector)
        
        return features

class SimpleWordEmbeddingExtractor:
    """简化的词嵌入特征提取器"""
    
    def __init__(self, embedding_dim=100):
        self.embedding_dim = embedding_dim
        self.word_vectors = {}
    
    def simple_tokenize(self, text):
        """简单分词"""
        if not text:
            return []
        words = re.findall(r'[\u4e00-\u9fff]+', text)
        return [word for word in words if len(word) >= 2]
    
    def train_simple_embeddings(self, texts):
        """训练简单的词嵌入（随机初始化，基于共现）"""
        print("训练简单词嵌入模型...")
        
        # 收集所有词汇
        all_words = set()
        for text in texts:
            words = self.simple_tokenize(text)
            all_words.update(words)
        
        # 为每个词生成随机向量（实际应用中应使用预训练向量）
        for word in all_words:
            # 使用词的哈希值作为种子，确保一致性
            random.seed(hash(word) % 10000)
            vector = [random.gauss(0, 0.1) for _ in range(self.embedding_dim)]
            self.word_vectors[word] = vector
        
        print(f"词嵌入模型训练完成，词汇量: {len(self.word_vectors)}")
    
    def get_text_embedding(self, text):
        """获取文本的平均词嵌入"""
        words = self.simple_tokenize(text)
        
        if not words:
            return [0.0] * self.embedding_dim
        
        # 获取词向量
        word_embeddings = []
        for word in words:
            if word in self.word_vectors:
                word_embeddings.append(self.word_vectors[word])
        
        if not word_embeddings:
            return [0.0] * self.embedding_dim
        
        # 计算平均嵌入
        avg_embedding = []
        for i in range(self.embedding_dim):
            avg_value = sum(embedding[i] for embedding in word_embeddings) / len(word_embeddings)
            avg_embedding.append(avg_value)
        
        return avg_embedding
    
    def extract_features(self, texts):
        """提取词嵌入特征"""
        if not self.word_vectors:
            self.train_simple_embeddings(texts)
        
        features = []
        for text in texts:
            embedding = self.get_text_embedding(text)
            features.append(embedding)
        
        return features

class SimpleSentimentExtractor:
    """简化的情感分析特征提取器"""
    
    def __init__(self):
        # 基础情感词典
        self.positive_words = {
            '好', '棒', '优秀', '完美', '杰出', '精彩', '美好', '幸福', '快乐',
            '开心', '高兴', '兴奋', '满意', '赞', '支持', '认同', '同意', '正确',
            '真实', '可靠', '值得', '信任', '安全', '健康', '有效', '成功',
            '希望', '乐观', '积极', '正面', '光明', '美丽', '漂亮', '可爱'
        }
        
        self.negative_words = {
            '坏', '差', '糟糕', '恶劣', '可怕', '恐怖', '危险', '有害', '毒', '假',
            '虚假', '不实', '谣言', '骗子', '欺骗', '诈骗', '威胁', '攻击',
            '伤害', '损害', '破坏', '污染', '错误', '失败', '失望', '绝望',
            '悲伤', '痛苦', '难过', '担心', '焦虑', '紧张', '害怕', '恐惧'
        }
        
        self.degree_words = {
            '非常': 2.0, '特别': 2.0, '十分': 2.0, '极其': 2.5, '极度': 2.5,
            '超级': 2.0, '绝对': 2.2, '完全': 2.0, '很': 1.5, '比较': 1.3,
            '相当': 1.4, '挺': 1.3, '还': 1.2, '有点': 1.1, '一点': 1.1
        }
        
        self.negation_words = {
            '不', '没', '无', '非', '未', '否', '别', '莫', '不是', '不会',
            '不能', '不要', '不用', '没有', '并非', '并不', '绝非', '决不'
        }
    
    def simple_tokenize(self, text):
        """简单分词"""
        if not text:
            return []
        words = re.findall(r'[\u4e00-\u9fff]+', text)
        return words
    
    def analyze_sentiment(self, text):
        """分析情感"""
        words = self.simple_tokenize(text)
        
        positive_score = 0
        negative_score = 0
        degree_modifier = 1.0
        negation_count = 0
        
        for word in words:
            if word in self.degree_words:
                degree_modifier = self.degree_words[word]
            elif word in self.negation_words:
                negation_count += 1
            elif word in self.positive_words:
                score = 1.0 * degree_modifier
                if negation_count % 2 == 1:
                    negative_score += score
                else:
                    positive_score += score
                degree_modifier = 1.0
            elif word in self.negative_words:
                score = 1.0 * degree_modifier
                if negation_count % 2 == 1:
                    positive_score += score
                else:
                    negative_score += score
                degree_modifier = 1.0
        
        # 计算特征
        total_score = positive_score - negative_score
        normalized_score = total_score / len(words) if words else 0
        sentiment_intensity = abs(normalized_score)
        
        # 统计特殊词汇
        urgency_words = {'紧急', '立即', '马上', '赶紧', '快', '速度', '急'}
        spread_words = {'转发', '扩散', '传播', '告诉', '通知', '分享', '转'}
        
        urgency_count = sum(1 for w in words if w in urgency_words)
        spread_count = sum(1 for w in words if w in spread_words)
        
        return [
            positive_score,
            negative_score,
            total_score,
            normalized_score,
            sentiment_intensity,
            sum(1 for w in words if w in self.positive_words),
            sum(1 for w in words if w in self.negative_words),
            urgency_count,
            spread_count
        ]
    
    def extract_features(self, texts):
        """提取情感特征"""
        features = []
        for text in texts:
            sentiment_features = self.analyze_sentiment(text)
            features.append(sentiment_features)
        return features

def load_sample_data():
    """加载示例数据"""
    try:
        # 尝试从JSON文件中读取少量数据
        with open('/workspace/rumors_v170613.json', 'r', encoding='utf-8') as f:
            texts = []
            labels = []
            count = 0
            
            for line in f:
                if count >= 50:  # 只读取50条数据作为演示
                    break
                
                try:
                    data = json.loads(line.strip())
                    text = data.get('rumorText', '')
                    title = data.get('title', '')
                    full_text = f"{title} {text}".strip()
                    
                    if full_text and len(full_text) > 10:
                        texts.append(full_text)
                        labels.append(1)  # 谣言
                        count += 1
                
                except json.JSONDecodeError:
                    continue
        
        # 添加一些非谣言样本
        non_rumor_texts = [
            "今天天气很好，适合出门运动，心情也很不错。",
            "刚刚吃了一顿美味的晚餐，感谢朋友的推荐。",
            "看了一部很棒的电影，剧情精彩，演技也很出色。",
            "工作进展顺利，感谢团队成员的支持和帮助。",
            "周末计划和家人聚会，准备做几道拿手菜。",
            "最近读了一本有趣的书，学到了很多新知识。",
            "参加了一个很有意义的活动，认识了新朋友。",
            "今天完成了一个重要的项目，感觉很有成就感。",
            "和朋友一起去爬山，欣赏了美丽的风景。",
            "学习了一门新技能，希望能在工作中应用。"
        ]
        
        texts.extend(non_rumor_texts)
        labels.extend([0] * len(non_rumor_texts))
        
        print(f"成功加载数据: {len(texts)} 条文本")
        print(f"谣言样本: {sum(labels)} 条")
        print(f"非谣言样本: {len(labels) - sum(labels)} 条")
        
        return texts, labels
    
    except FileNotFoundError:
        print("数据文件未找到，使用预设示例数据...")
        
        # 预设示例数据
        texts = [
            "紧急扩散！银行系统明天将全面崩溃，大家赶紧去取钱！这是内部消息！",
            "今天天气很好，和朋友一起去公园散步，心情很愉快。",
            "据权威专家最新研究发现，这种食品含有大量致癌物质，千万不要食用！",
            "刚刚看了一部很棒的电影，剧情精彩，演技也很出色，推荐给大家。",
            "注意！新型病毒正在快速传播，症状包括发热咳嗽，请大家立即做好防护！",
            "周末计划和家人聚餐，准备做几道拿手菜，很期待这次聚会。",
            "震惊！科学家发现外星人信号，政府正在隐瞒真相！速转！",
            "今天学习了新的编程技术，感觉很有收获，分享给同事们。",
            "重大消息！所有手机将在下周停止服务，运营商内部通知！",
            "和朋友一起参加了音乐会，演出很精彩，度过了愉快的夜晚。"
        ]
        
        labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=谣言, 0=非谣言
        
        return texts, labels

def demo_feature_extraction():
    """演示三种特征提取方法"""
    
    print("谣言检测特征提取演示")
    print("=" * 60)
    
    # 1. 加载数据
    texts, labels = load_sample_data()
    
    # 2. TF-IDF特征提取
    print("\n1. TF-IDF特征提取")
    print("-" * 40)
    
    tfidf_extractor = SimpleTFIDFExtractor(max_features=500)
    tfidf_features = tfidf_extractor.extract_features(texts)
    
    print(f"TF-IDF特征维度: {len(tfidf_features[0])}")
    print(f"样本1前10个特征值: {tfidf_features[0][:10]}")
    
    # 3. 词嵌入特征提取
    print("\n2. 词嵌入特征提取")
    print("-" * 40)
    
    embedding_extractor = SimpleWordEmbeddingExtractor(embedding_dim=50)
    embedding_features = embedding_extractor.extract_features(texts)
    
    print(f"词嵌入特征维度: {len(embedding_features[0])}")
    print(f"样本1前10个特征值: {embedding_features[0][:10]}")
    
    # 4. 情感分析特征提取
    print("\n3. 情感分析特征提取")
    print("-" * 40)
    
    sentiment_extractor = SimpleSentimentExtractor()
    sentiment_features = sentiment_extractor.extract_features(texts)
    
    print(f"情感分析特征维度: {len(sentiment_features[0])}")
    print(f"特征名称: ['positive_score', 'negative_score', 'total_score', 'normalized_score', 'sentiment_intensity', 'positive_word_count', 'negative_word_count', 'urgency_count', 'spread_count']")
    
    # 5. 特征分析
    print("\n4. 特征分析")
    print("-" * 40)
    
    for i in range(min(5, len(texts))):
        text = texts[i]
        label = labels[i]
        
        print(f"\n样本 {i+1} ({'谣言' if label == 1 else '非谣言'}):")
        print(f"文本: {text[:60]}...")
        print(f"TF-IDF特征均值: {sum(tfidf_features[i])/len(tfidf_features[i]):.4f}")
        print(f"词嵌入特征均值: {sum(embedding_features[i])/len(embedding_features[i]):.4f}")
        print(f"情感分数: {sentiment_features[i][3]:.4f}")
        print(f"情感强度: {sentiment_features[i][4]:.4f}")
        print(f"紧急性指标: {sentiment_features[i][7]}")
        print(f"传播性指标: {sentiment_features[i][8]}")
    
    # 6. 合并特征
    print("\n5. 特征合并")
    print("-" * 40)
    
    combined_features = []
    for i in range(len(texts)):
        combined = tfidf_features[i] + embedding_features[i] + sentiment_features[i]
        combined_features.append(combined)
    
    total_features = len(combined_features[0])
    print(f"合并后特征维度: {total_features}")
    print(f"特征组成: TF-IDF({len(tfidf_features[0])}) + 词嵌入({len(embedding_features[0])}) + 情感分析({len(sentiment_features[0])})")
    
    # 7. 保存结果
    print("\n6. 保存结果")
    print("-" * 40)
    
    # 保存为简单的文本格式
    with open('/workspace/demo_features.txt', 'w', encoding='utf-8') as f:
        f.write("谣言检测特征提取结果\n")
        f.write("=" * 50 + "\n\n")
        
        for i, (text, label) in enumerate(zip(texts, labels)):
            f.write(f"样本 {i+1} ({'谣言' if label == 1 else '非谣言'}):\n")
            f.write(f"文本: {text}\n")
            f.write(f"TF-IDF特征: {tfidf_features[i][:5]}...\n")
            f.write(f"词嵌入特征: {embedding_features[i][:5]}...\n")
            f.write(f"情感特征: {sentiment_features[i]}\n")
            f.write("-" * 50 + "\n")
    
    print("结果已保存到 demo_features.txt")
    
    # 8. 简单的分类评估
    print("\n7. 简单分类评估")
    print("-" * 40)
    
    # 使用情感分数作为简单分类器
    correct_predictions = 0
    for i, label in enumerate(labels):
        sentiment_score = sentiment_features[i][3]  # normalized_score
        urgency_score = sentiment_features[i][7]
        spread_score = sentiment_features[i][8]
        
        # 简单规则：负面情感 + 高紧急性/传播性 = 谣言
        predicted_rumor = (sentiment_score < -0.1) or (urgency_score > 0) or (spread_score > 0)
        predicted_label = 1 if predicted_rumor else 0
        
        if predicted_label == label:
            correct_predictions += 1
    
    accuracy = correct_predictions / len(labels)
    print(f"简单规则分类准确率: {accuracy:.3f} ({correct_predictions}/{len(labels)})")
    
    return {
        'texts': texts,
        'labels': labels,
        'tfidf_features': tfidf_features,
        'embedding_features': embedding_features,
        'sentiment_features': sentiment_features,
        'combined_features': combined_features
    }

if __name__ == "__main__":
    # 运行演示
    results = demo_feature_extraction()
    
    print("\n" + "=" * 60)
    print("特征提取演示完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("- demo_features.txt: 详细的特征提取结果")
    print("\n特征提取器说明:")
    print("1. TF-IDF特征: 基于词频-逆文档频率，谣言关键词加权")
    print("2. 词嵌入特征: 基于词向量的平均嵌入表示")
    print("3. 情感分析特征: 多维情感指标，包含谣言特定特征")
    print("\n注意: 这是简化版本，完整版本请安装依赖后使用其他脚本。")