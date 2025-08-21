"""
词嵌入特征提取器
支持Word2Vec, GloVe, BERT等多种预训练词向量
"""

import numpy as np
import jieba
from gensim.models import Word2Vec
from transformers import AutoTokenizer, AutoModel
import torch
import warnings
warnings.filterwarnings('ignore')

class WordEmbeddingExtractor:
    """
    词嵌入特征提取器
    支持多种词嵌入方法：Word2Vec, BERT, 平均词向量等
    """
    
    def __init__(self, method='word2vec', embedding_dim=300, model_path=None):
        """
        初始化词嵌入提取器
        
        Args:
            method: 嵌入方法 ('word2vec', 'bert', 'glove', 'fasttext')
            embedding_dim: 嵌入维度
            model_path: 预训练模型路径
        """
        self.method = method
        self.embedding_dim = embedding_dim
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        # 初始化jieba
        jieba.initialize()
        
        # 根据方法初始化模型
        self._init_model()
    
    def _init_model(self):
        """初始化嵌入模型"""
        if self.method == 'bert':
            self._init_bert_model()
        elif self.method == 'word2vec':
            self._init_word2vec_model()
        else:
            print(f"警告: {self.method} 方法需要手动加载模型")
    
    def _init_bert_model(self):
        """初始化BERT模型"""
        try:
            # 使用中文BERT模型
            model_name = self.model_path or 'bert-base-chinese'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            print(f"成功加载BERT模型: {model_name}")
        except Exception as e:
            print(f"BERT模型加载失败: {e}")
            print("将使用随机初始化的嵌入")
            self.method = 'random'
    
    def _init_word2vec_model(self):
        """初始化Word2Vec模型"""
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model = Word2Vec.load(self.model_path)
                self.embedding_dim = self.model.vector_size
                print(f"成功加载Word2Vec模型: {self.model_path}")
            except Exception as e:
                print(f"Word2Vec模型加载失败: {e}")
                self.model = None
        else:
            print("未提供Word2Vec模型路径，将在训练时创建新模型")
    
    def _preprocess_text(self, text):
        """文本预处理和分词"""
        if not text or not isinstance(text, str):
            return []
        
        # 清理文本
        import re
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'@[^\s]+', '', text)
        text = re.sub(r'#[^#]+#', '', text)
        
        # 分词
        words = list(jieba.cut(text))
        
        # 过滤空词和单字符词
        words = [word.strip() for word in words if len(word.strip()) > 1]
        
        return words
    
    def _get_bert_embedding(self, text):
        """获取BERT嵌入"""
        try:
            # 编码文本
            inputs = self.tokenizer(text, return_tensors='pt', 
                                  truncation=True, max_length=512, 
                                  padding=True)
            
            # 获取嵌入
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用[CLS]标记的嵌入作为句子表示
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            
            return embedding
        except Exception as e:
            print(f"BERT嵌入提取失败: {e}")
            return np.random.normal(0, 0.1, self.embedding_dim)
    
    def _get_word2vec_embedding(self, words):
        """获取Word2Vec平均嵌入"""
        if self.model is None:
            return np.random.normal(0, 0.1, self.embedding_dim)
        
        embeddings = []
        for word in words:
            try:
                if word in self.model.wv:
                    embeddings.append(self.model.wv[word])
            except:
                continue
        
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.random.normal(0, 0.1, self.embedding_dim)
    
    def _get_average_embedding(self, words):
        """获取平均词嵌入（使用随机初始化的词向量）"""
        # 这里可以替换为预训练的词向量
        embeddings = []
        for word in words:
            # 简单的词向量模拟（实际应用中应使用预训练向量）
            word_hash = hash(word) % 10000
            np.random.seed(word_hash)
            embedding = np.random.normal(0, 0.1, self.embedding_dim)
            embeddings.append(embedding)
        
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(self.embedding_dim)
    
    def train_word2vec(self, texts, vector_size=300, window=5, min_count=2, workers=4):
        """
        训练Word2Vec模型
        
        Args:
            texts: 文本列表
            vector_size: 向量维度
            window: 窗口大小
            min_count: 最小词频
            workers: 线程数
        """
        # 预处理所有文本
        sentences = [self._preprocess_text(text) for text in texts]
        sentences = [words.split() for words in sentences if words]
        
        # 训练Word2Vec模型
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=1  # 使用Skip-gram
        )
        
        self.embedding_dim = vector_size
        print(f"Word2Vec模型训练完成，词汇量: {len(self.model.wv)}")
    
    def get_text_embedding(self, text):
        """
        获取单个文本的嵌入向量
        
        Args:
            text: 输入文本
        
        Returns:
            embedding: 嵌入向量 [embedding_dim]
        """
        if self.method == 'bert':
            return self._get_bert_embedding(text)
        elif self.method == 'word2vec':
            words = self._preprocess_text(text)
            return self._get_word2vec_embedding(words)
        else:
            words = self._preprocess_text(text)
            return self._get_average_embedding(words)
    
    def extract_features(self, texts):
        """
        批量提取文本嵌入特征
        
        Args:
            texts: 文本列表
        
        Returns:
            embeddings: 嵌入矩阵 [N, embedding_dim]
        """
        embeddings = []
        
        print(f"正在提取 {len(texts)} 个文本的 {self.method} 嵌入特征...")
        
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                print(f"进度: {i}/{len(texts)}")
            
            embedding = self.get_text_embedding(text)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        print(f"词嵌入特征提取完成，特征矩阵形状: {embeddings.shape}")
        
        return embeddings
    
    def save_model(self, filepath):
        """保存模型"""
        if self.method == 'word2vec' and self.model:
            self.model.save(filepath)
        else:
            print(f"警告: {self.method} 模型无法保存或不需要保存")
    
    def load_word2vec_model(self, filepath):
        """加载Word2Vec模型"""
        try:
            self.model = Word2Vec.load(filepath)
            self.embedding_dim = self.model.vector_size
            print(f"成功加载Word2Vec模型: {filepath}")
        except Exception as e:
            print(f"Word2Vec模型加载失败: {e}")

def extract_word_embedding_features(texts, method='word2vec', embedding_dim=300, 
                                  model_path=None, train_new_model=False):
    """
    便捷函数：提取词嵌入特征
    
    Args:
        texts: 文本列表
        method: 嵌入方法 ('word2vec', 'bert', 'average')
        embedding_dim: 嵌入维度
        model_path: 预训练模型路径
        train_new_model: 是否训练新的Word2Vec模型
    
    Returns:
        embeddings: 词嵌入特征矩阵 [N, embedding_dim]
        extractor: 特征提取器
    """
    extractor = WordEmbeddingExtractor(
        method=method, 
        embedding_dim=embedding_dim, 
        model_path=model_path
    )
    
    # 如果是Word2Vec且需要训练新模型
    if method == 'word2vec' and train_new_model:
        extractor.train_word2vec(texts, vector_size=embedding_dim)
    
    # 提取特征
    embeddings = extractor.extract_features(texts)
    
    return embeddings, extractor

# 使用示例
if __name__ == "__main__":
    # 示例文本
    sample_texts = [
        "紧急扩散！这个消息一定要转发给所有人看到！",
        "今天天气很好，适合出门散步。",
        "据专家研究表明，这种食品含有致癌物质，大家千万不要吃！",
        "刚刚看到一个有趣的新闻，分享给大家。"
    ]
    
    # 方法1：使用Word2Vec（训练新模型）
    print("=== Word2Vec嵌入 ===")
    embeddings_w2v, extractor_w2v = extract_word_embedding_features(
        sample_texts, method='word2vec', embedding_dim=100, train_new_model=True
    )
    
    # 方法2：使用BERT（需要安装transformers）
    print("\n=== BERT嵌入 ===")
    try:
        embeddings_bert, extractor_bert = extract_word_embedding_features(
            sample_texts, method='bert'
        )
    except Exception as e:
        print(f"BERT嵌入失败: {e}")
        print("请安装transformers库: pip install transformers torch")
    
    # 方法3：使用平均词向量
    print("\n=== 平均词向量嵌入 ===")
    embeddings_avg, extractor_avg = extract_word_embedding_features(
        sample_texts, method='average', embedding_dim=300
    )