"""
谣言检测特征提取使用示例
演示如何使用三种特征提取器
"""

import numpy as np
from data_loader import load_and_preprocess_data
from tfidf_feature_extractor import extract_tfidf_features
from word_embedding_extractor import extract_word_embedding_features
from sentiment_feature_extractor import extract_sentiment_features

def demo_feature_extraction():
    """演示特征提取过程"""
    
    print("谣言检测特征提取演示")
    print("=" * 50)
    
    # 1. 加载数据
    print("1. 加载数据...")
    texts, labels, metadata, data_info = load_and_preprocess_data(
        max_samples=100,  # 小样本演示
        balance=True
    )
    
    if not texts:
        print("无法加载数据，使用示例数据...")
        texts = [
            "紧急扩散！银行系统将在明天崩溃，大家赶紧去取钱！这是内部消息！",
            "今天天气很好，和朋友一起去公园散步，心情很愉快。",
            "据权威专家研究发现，这种食品含有大量致癌物质，千万不要食用！",
            "刚刚看了一部很棒的电影，剧情精彩，演技也很出色，推荐给大家。",
            "注意！新型病毒正在快速传播，症状包括发热咳嗽，请大家做好防护！",
            "周末计划和家人聚餐，准备做几道拿手菜，很期待这次聚会。"
        ]
        labels = [1, 0, 1, 0, 1, 0]  # 1=谣言, 0=非谣言
    
    print(f"使用 {len(texts)} 个文本样本进行演示")
    
    # 2. 特征提取演示
    print("\n" + "=" * 50)
    print("2. 特征提取")
    print("=" * 50)
    
    # 2.1 TF-IDF特征
    print("\n2.1 改进的TF-IDF特征提取")
    print("-" * 30)
    tfidf_features, tfidf_extractor = extract_tfidf_features(texts, max_features=1000)
    
    print(f"TF-IDF特征矩阵形状: {tfidf_features.shape}")
    print(f"前5个特征名称: {tfidf_extractor.get_feature_names()[:5]}")
    print(f"样本1的前10个TF-IDF特征值: {tfidf_features[0][:10]}")
    
    # 2.2 词嵌入特征
    print("\n2.2 词嵌入特征提取")
    print("-" * 30)
    embedding_features, embedding_extractor = extract_word_embedding_features(
        texts, method='word2vec', embedding_dim=100, train_new_model=True
    )
    
    print(f"词嵌入特征矩阵形状: {embedding_features.shape}")
    print(f"样本1的前10个词嵌入特征值: {embedding_features[0][:10]}")
    
    # 2.3 情感分析特征
    print("\n2.3 情感分析特征提取")
    print("-" * 30)
    sentiment_features, sentiment_extractor, sentiment_names = extract_sentiment_features(texts)
    
    print(f"情感分析特征矩阵形状: {sentiment_features.shape}")
    print(f"特征名称: {sentiment_names}")
    print(f"样本1的情感特征: {sentiment_features[0]}")
    
    # 3. 特征分析
    print("\n" + "=" * 50)
    print("3. 特征分析")
    print("=" * 50)
    
    for i, (text, label) in enumerate(zip(texts[:3], labels[:3])):
        print(f"\n样本 {i+1} ({'谣言' if label == 1 else '非谣言'}):")
        print(f"文本: {text[:80]}...")
        print(f"TF-IDF特征统计: 均值={np.mean(tfidf_features[i]):.4f}, 最大值={np.max(tfidf_features[i]):.4f}")
        print(f"词嵌入特征统计: 均值={np.mean(embedding_features[i]):.4f}, 标准差={np.std(embedding_features[i]):.4f}")
        print(f"情感分析分数: {sentiment_features[i][3]:.4f}")  # normalized_score
        print(f"情感强度: {sentiment_features[i][4]:.4f}")      # sentiment_intensity
    
    # 4. 合并特征
    print("\n4. 合并所有特征")
    print("-" * 30)
    combined_features = np.hstack([tfidf_features, embedding_features, sentiment_features])
    print(f"合并特征矩阵形状: {combined_features.shape}")
    print(f"特征维度分布: TF-IDF({tfidf_features.shape[1]}) + 词嵌入({embedding_features.shape[1]}) + 情感分析({sentiment_features.shape[1]}) = {combined_features.shape[1]}")
    
    # 5. 保存演示结果
    print("\n5. 保存演示结果")
    print("-" * 30)
    
    import pandas as pd
    
    # 创建DataFrame
    df = pd.DataFrame({
        'text': texts,
        'label': labels,
        'tfidf_mean': np.mean(tfidf_features, axis=1),
        'embedding_mean': np.mean(embedding_features, axis=1),
        'sentiment_score': sentiment_features[:, 3],  # normalized_score
        'sentiment_intensity': sentiment_features[:, 4]  # sentiment_intensity
    })
    
    # 保存到CSV
    df.to_csv("demo_results.csv", index=False, encoding='utf-8')
    print("演示结果已保存到 demo_results.csv")
    
    # 保存完整特征矩阵
    np.save("demo_combined_features.npy", combined_features)
    print("完整特征矩阵已保存到 demo_combined_features.npy")
    
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
    
    print("\n" + "=" * 50)
    print("演示完成！")
    print("=" * 50)
    print("\n生成的文件:")
    print("- demo_results.csv: 演示结果摘要")
    print("- demo_combined_features.npy: 完整特征矩阵")
    print("\n可以使用以下命令运行完整训练:")
    print("python rumor_detection_features.py")
    print("\n或者只提取特征:")
    print("python rumor_detection_features.py --extract-only")