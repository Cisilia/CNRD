"""
谣言检测特征提取主程序
整合三种特征提取器：TF-IDF、词嵌入、情感分析
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from tfidf_feature_extractor import extract_tfidf_features
from word_embedding_extractor import extract_word_embedding_features
from sentiment_feature_extractor import extract_sentiment_features
from data_loader import load_and_preprocess_data, split_dataset

class RumorDetectionFeatureExtractor:
    """
    谣言检测特征提取器
    整合三种特征：TF-IDF、词嵌入、情感分析
    """
    
    def __init__(self, tfidf_max_features=5000, embedding_dim=300, embedding_method='word2vec'):
        self.tfidf_max_features = tfidf_max_features
        self.embedding_dim = embedding_dim
        self.embedding_method = embedding_method
        
        # 特征提取器
        self.tfidf_extractor = None
        self.embedding_extractor = None
        self.sentiment_extractor = None
        
        # 特征缩放器
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def extract_all_features(self, texts, train_new_embedding=True):
        """
        提取所有三种特征
        
        Args:
            texts: 文本列表
            train_new_embedding: 是否训练新的词嵌入模型
        
        Returns:
            combined_features: 合并的特征矩阵
            feature_info: 特征信息
        """
        print("=" * 50)
        print("开始提取谣言检测特征")
        print("=" * 50)
        
        # 1. 提取TF-IDF特征
        print("\n1. 提取改进的TF-IDF特征...")
        tfidf_features, self.tfidf_extractor = extract_tfidf_features(
            texts, max_features=self.tfidf_max_features
        )
        
        # 2. 提取词嵌入特征
        print("\n2. 提取词嵌入特征...")
        embedding_features, self.embedding_extractor = extract_word_embedding_features(
            texts, 
            method=self.embedding_method,
            embedding_dim=self.embedding_dim,
            train_new_model=train_new_embedding
        )
        
        # 3. 提取情感分析特征
        print("\n3. 提取情感分析特征...")
        sentiment_features, self.sentiment_extractor, sentiment_names = extract_sentiment_features(texts)
        
        # 4. 合并特征
        print("\n4. 合并特征...")
        combined_features = np.hstack([
            tfidf_features,
            embedding_features,
            sentiment_features
        ])
        
        # 构建特征名称
        tfidf_names = [f"tfidf_{i}" for i in range(tfidf_features.shape[1])]
        embedding_names = [f"embedding_{i}" for i in range(embedding_features.shape[1])]
        
        self.feature_names = tfidf_names + embedding_names + sentiment_names
        
        # 特征信息
        feature_info = {
            'total_features': combined_features.shape[1],
            'tfidf_features': tfidf_features.shape[1],
            'embedding_features': embedding_features.shape[1],
            'sentiment_features': sentiment_features.shape[1],
            'feature_names': self.feature_names
        }
        
        print(f"\n特征提取完成:")
        print(f"- TF-IDF特征: {feature_info['tfidf_features']} 维")
        print(f"- 词嵌入特征: {feature_info['embedding_features']} 维")
        print(f"- 情感分析特征: {feature_info['sentiment_features']} 维")
        print(f"- 总特征数: {feature_info['total_features']} 维")
        print(f"- 特征矩阵形状: {combined_features.shape}")
        
        return combined_features, feature_info
    
    def fit_scaler(self, features):
        """训练特征缩放器"""
        self.scaler.fit(features)
    
    def transform_features(self, features):
        """标准化特征"""
        return self.scaler.transform(features)
    
    def save_extractors(self, save_dir="models"):
        """保存所有特征提取器"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存TF-IDF提取器
        if self.tfidf_extractor:
            tfidf_path = os.path.join(save_dir, "tfidf_extractor.pkl")
            self.tfidf_extractor.save_model(tfidf_path)
            print(f"TF-IDF提取器已保存: {tfidf_path}")
        
        # 保存词嵌入模型
        if self.embedding_extractor:
            embedding_path = os.path.join(save_dir, "word2vec_model.model")
            self.embedding_extractor.save_model(embedding_path)
            print(f"词嵌入模型已保存: {embedding_path}")
        
        # 保存缩放器
        scaler_path = os.path.join(save_dir, "feature_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"特征缩放器已保存: {scaler_path}")

def build_rumor_detection_model(texts, labels, test_size=0.2):
    """
    构建完整的谣言检测模型
    
    Args:
        texts: 文本列表
        labels: 标签列表
        test_size: 测试集比例
    
    Returns:
        model: 训练好的分类器
        feature_extractor: 特征提取器
        test_results: 测试结果
    """
    print("=" * 60)
    print("构建谣言检测模型")
    print("=" * 60)
    
    # 1. 划分数据集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # 2. 初始化特征提取器
    feature_extractor = RumorDetectionFeatureExtractor(
        tfidf_max_features=3000,
        embedding_dim=200,
        embedding_method='word2vec'
    )
    
    # 3. 提取训练集特征
    print("\n提取训练集特征...")
    train_features, feature_info = feature_extractor.extract_all_features(
        train_texts, train_new_embedding=True
    )
    
    # 4. 标准化特征
    feature_extractor.fit_scaler(train_features)
    train_features_scaled = feature_extractor.transform_features(train_features)
    
    # 5. 训练分类器
    print("\n训练随机森林分类器...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(train_features_scaled, train_labels)
    
    # 6. 测试模型
    print("\n提取测试集特征...")
    # 对测试集提取特征（不重新训练提取器）
    test_tfidf = feature_extractor.tfidf_extractor.transform(test_texts)
    test_embedding = feature_extractor.embedding_extractor.extract_features(test_texts)
    test_sentiment = feature_extractor.sentiment_extractor.extract_features(test_texts)
    
    test_features = np.hstack([test_tfidf, test_embedding, test_sentiment])
    test_features_scaled = feature_extractor.transform_features(test_features)
    
    # 7. 预测和评估
    test_predictions = model.predict(test_features_scaled)
    test_probabilities = model.predict_proba(test_features_scaled)
    
    print("\n模型评估结果:")
    print(classification_report(test_labels, test_predictions, 
                              target_names=['非谣言', '谣言']))
    
    # 8. 特征重要性分析
    feature_importance = model.feature_importances_
    
    # 获取最重要的特征
    top_features_idx = np.argsort(feature_importance)[-20:]
    print("\n最重要的20个特征:")
    for i, idx in enumerate(reversed(top_features_idx)):
        feature_name = feature_extractor.feature_names[idx] if idx < len(feature_extractor.feature_names) else f"feature_{idx}"
        print(f"{i+1:2d}. {feature_name}: {feature_importance[idx]:.4f}")
    
    test_results = {
        'predictions': test_predictions,
        'probabilities': test_probabilities,
        'true_labels': test_labels,
        'feature_importance': feature_importance,
        'classification_report': classification_report(test_labels, test_predictions, output_dict=True)
    }
    
    return model, feature_extractor, test_results

def predict_rumor(text, model, feature_extractor):
    """
    预测单个文本是否为谣言
    
    Args:
        text: 输入文本
        model: 训练好的分类器
        feature_extractor: 特征提取器
    
    Returns:
        prediction: 预测结果 (0=非谣言, 1=谣言)
        probability: 预测概率 [非谣言概率, 谣言概率]
        features: 提取的特征
    """
    # 提取特征
    tfidf_features = feature_extractor.tfidf_extractor.transform([text])
    embedding_features = feature_extractor.embedding_extractor.extract_features([text])
    sentiment_features = feature_extractor.sentiment_extractor.extract_features([text])
    
    # 合并特征
    combined_features = np.hstack([tfidf_features, embedding_features, sentiment_features])
    
    # 标准化
    features_scaled = feature_extractor.transform_features(combined_features)
    
    # 预测
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return prediction, probability, combined_features[0]

# 主程序
def main():
    """主程序：完整的谣言检测特征提取和模型训练流程"""
    
    print("谣言检测特征提取系统")
    print("=" * 60)
    
    # 1. 加载数据
    print("1. 加载数据...")
    texts, labels, metadata, data_info = load_and_preprocess_data(
        max_samples=2000,  # 可以根据需要调整
        balance=True,
        min_text_length=10
    )
    
    if not texts:
        print("错误：无法加载数据！")
        return
    
    # 2. 构建模型
    model, feature_extractor, test_results = build_rumor_detection_model(texts, labels)
    
    # 3. 保存模型和特征提取器
    print("\n保存模型和特征提取器...")
    feature_extractor.save_extractors("models")
    
    model_path = "models/rumor_classifier.pkl"
    os.makedirs("models", exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"分类器已保存: {model_path}")
    
    # 4. 演示预测
    print("\n" + "=" * 60)
    print("演示谣言检测")
    print("=" * 60)
    
    demo_texts = [
        "紧急通知！明天所有银行将关闭，大家赶紧去取钱！",
        "今天天气很好，适合出门散步，心情也很不错。",
        "据权威专家证实，这种新药有严重副作用，千万不要使用！",
        "刚刚看了一部很棒的电影，推荐给大家观看。"
    ]
    
    for i, text in enumerate(demo_texts):
        prediction, probability, features = predict_rumor(text, model, feature_extractor)
        
        print(f"\n文本 {i+1}: {text}")
        print(f"预测结果: {'谣言' if prediction == 1 else '非谣言'}")
        print(f"置信度: 非谣言={probability[0]:.3f}, 谣言={probability[1]:.3f}")
    
    print("\n" + "=" * 60)
    print("特征提取完成！")
    print("=" * 60)
    
    return model, feature_extractor, test_results

# 独立的特征提取函数
def extract_rumor_features(texts, save_features=True, output_dir="features"):
    """
    独立的特征提取函数
    只提取特征，不训练分类器
    
    Args:
        texts: 文本列表
        save_features: 是否保存特征到文件
        output_dir: 输出目录
    
    Returns:
        all_features: 所有特征的字典
    """
    print("=" * 50)
    print("独立特征提取模式")
    print("=" * 50)
    
    all_features = {}
    
    # 1. TF-IDF特征
    print("\n提取TF-IDF特征...")
    tfidf_features, tfidf_extractor = extract_tfidf_features(texts, max_features=5000)
    all_features['tfidf'] = tfidf_features
    
    # 2. 词嵌入特征
    print("\n提取词嵌入特征...")
    embedding_features, embedding_extractor = extract_word_embedding_features(
        texts, method='word2vec', embedding_dim=300, train_new_model=True
    )
    all_features['embedding'] = embedding_features
    
    # 3. 情感分析特征
    print("\n提取情感分析特征...")
    sentiment_features, sentiment_extractor, sentiment_names = extract_sentiment_features(texts)
    all_features['sentiment'] = sentiment_features
    
    # 4. 合并所有特征
    combined_features = np.hstack([tfidf_features, embedding_features, sentiment_features])
    all_features['combined'] = combined_features
    
    print(f"\n特征提取汇总:")
    print(f"- 文本数量: {len(texts)}")
    print(f"- TF-IDF特征: {tfidf_features.shape}")
    print(f"- 词嵌入特征: {embedding_features.shape}")
    print(f"- 情感分析特征: {sentiment_features.shape}")
    print(f"- 合并特征: {combined_features.shape}")
    
    # 5. 保存特征（可选）
    if save_features:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为numpy文件
        np.save(os.path.join(output_dir, "tfidf_features.npy"), tfidf_features)
        np.save(os.path.join(output_dir, "embedding_features.npy"), embedding_features)
        np.save(os.path.join(output_dir, "sentiment_features.npy"), sentiment_features)
        np.save(os.path.join(output_dir, "combined_features.npy"), combined_features)
        
        # 保存为CSV文件（便于查看）
        df_tfidf = pd.DataFrame(tfidf_features, columns=[f"tfidf_{i}" for i in range(tfidf_features.shape[1])])
        df_embedding = pd.DataFrame(embedding_features, columns=[f"embedding_{i}" for i in range(embedding_features.shape[1])])
        df_sentiment = pd.DataFrame(sentiment_features, columns=sentiment_names)
        
        df_tfidf.to_csv(os.path.join(output_dir, "tfidf_features.csv"), index=False)
        df_embedding.to_csv(os.path.join(output_dir, "embedding_features.csv"), index=False)
        df_sentiment.to_csv(os.path.join(output_dir, "sentiment_features.csv"), index=False)
        
        # 合并所有特征的DataFrame
        df_combined = pd.concat([df_tfidf, df_embedding, df_sentiment], axis=1)
        df_combined.to_csv(os.path.join(output_dir, "all_features.csv"), index=False)
        
        print(f"\n特征已保存到目录: {output_dir}")
    
    return all_features

if __name__ == "__main__":
    # 检查命令行参数，决定运行模式
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--extract-only":
        # 只提取特征模式
        print("运行模式：仅特征提取")
        
        # 加载数据
        texts, labels, metadata, data_info = load_and_preprocess_data(
            max_samples=1000,  # 可根据需要调整
            balance=True
        )
        
        if texts:
            # 提取特征
            features = extract_rumor_features(texts, save_features=True)
            print("\n特征提取完成！特征文件已保存到 'features' 目录。")
        else:
            print("错误：无法加载数据！")
    
    else:
        # 完整模式：特征提取 + 模型训练
        print("运行模式：完整训练")
        
        # 加载数据
        texts, labels, metadata, data_info = load_and_preprocess_data(
            max_samples=1000,  # 可根据需要调整
            balance=True
        )
        
        if texts:
            # 构建模型
            model, feature_extractor, test_results = main()
        else:
            print("错误：无法加载数据！")
    
    print("\n程序执行完成！")