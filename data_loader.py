"""
数据加载和预处理模块
支持加载中文谣言数据集并进行预处理
"""

import json
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import re
from collections import Counter

class RumorDataLoader:
    """
    谣言数据集加载器
    支持加载两种数据集格式：
    1. rumors_v170613.json - 基础谣言数据
    2. CED_Dataset - 包含转发评论的完整数据
    """
    
    def __init__(self, data_dir="/workspace"):
        self.data_dir = data_dir
        self.rumors_file = os.path.join(data_dir, "rumors_v170613.json")
        self.ced_dir = os.path.join(data_dir, "CED_Dataset")
    
    def load_basic_rumors(self, max_samples=None):
        """
        加载基础谣言数据集
        
        Args:
            max_samples: 最大样本数，None表示加载全部
        
        Returns:
            texts: 文本列表
            labels: 标签列表 (1表示谣言，0表示非谣言)
            metadata: 元数据列表
        """
        texts = []
        labels = []
        metadata = []
        
        print(f"正在加载基础谣言数据集: {self.rumors_file}")
        
        try:
            with open(self.rumors_file, 'r', encoding='utf-8') as f:
                count = 0
                for line in f:
                    if max_samples and count >= max_samples:
                        break
                    
                    try:
                        data = json.loads(line.strip())
                        
                        # 提取文本内容
                        text = data.get('rumorText', '')
                        title = data.get('title', '')
                        
                        # 合并标题和内容
                        full_text = f"{title} {text}".strip()
                        
                        if full_text:
                            texts.append(full_text)
                            labels.append(1)  # 所有数据都是谣言
                            
                            # 保存元数据
                            metadata.append({
                                'rumorCode': data.get('rumorCode', ''),
                                'publishTime': data.get('publishTime', ''),
                                'visitTimes': data.get('visitTimes', 0),
                                'result': data.get('result', '')
                            })
                            
                            count += 1
                    
                    except json.JSONDecodeError:
                        continue
        
        except FileNotFoundError:
            print(f"文件不存在: {self.rumors_file}")
            return [], [], []
        
        print(f"成功加载 {len(texts)} 条谣言数据")
        return texts, labels, metadata
    
    def load_ced_dataset(self, max_samples=None):
        """
        加载CED数据集（包含谣言和非谣言）
        
        Args:
            max_samples: 每类最大样本数
        
        Returns:
            texts: 文本列表
            labels: 标签列表 (1表示谣言，0表示非谣言)
            metadata: 元数据列表
        """
        texts = []
        labels = []
        metadata = []
        
        # 加载原微博数据
        original_dir = os.path.join(self.ced_dir, "original-microblog")
        rumor_repost_dir = os.path.join(self.ced_dir, "rumor-repost")
        non_rumor_repost_dir = os.path.join(self.ced_dir, "non-rumor-repost")
        
        if not os.path.exists(original_dir):
            print(f"CED数据集目录不存在: {self.ced_dir}")
            return [], [], []
        
        print("正在加载CED数据集...")
        
        # 获取谣言和非谣言文件列表
        rumor_files = set()
        non_rumor_files = set()
        
        if os.path.exists(rumor_repost_dir):
            rumor_files = {f.split('_')[0] for f in os.listdir(rumor_repost_dir) if f.endswith('.json')}
        
        if os.path.exists(non_rumor_repost_dir):
            non_rumor_files = {f.split('_')[0] for f in os.listdir(non_rumor_repost_dir) if f.endswith('.json')}
        
        # 加载原微博文件
        original_files = [f for f in os.listdir(original_dir) if f.endswith('.json')]
        
        rumor_count = 0
        non_rumor_count = 0
        
        for filename in original_files:
            file_id = filename.split('_')[0]
            filepath = os.path.join(original_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                text = data.get('text', '')
                if not text:
                    continue
                
                # 判断是否为谣言
                is_rumor = file_id in rumor_files
                is_non_rumor = file_id in non_rumor_files
                
                if is_rumor and (max_samples is None or rumor_count < max_samples):
                    texts.append(text)
                    labels.append(1)  # 谣言
                    metadata.append({
                        'file_id': file_id,
                        'filename': filename,
                        'user': data.get('user', ''),
                        'time': data.get('time', ''),
                        'type': 'rumor'
                    })
                    rumor_count += 1
                
                elif is_non_rumor and (max_samples is None or non_rumor_count < max_samples):
                    texts.append(text)
                    labels.append(0)  # 非谣言
                    metadata.append({
                        'file_id': file_id,
                        'filename': filename,
                        'user': data.get('user', ''),
                        'time': data.get('time', ''),
                        'type': 'non_rumor'
                    })
                    non_rumor_count += 1
            
            except Exception as e:
                print(f"加载文件失败 {filename}: {e}")
                continue
        
        print(f"CED数据集加载完成:")
        print(f"- 谣言样本: {rumor_count}")
        print(f"- 非谣言样本: {non_rumor_count}")
        print(f"- 总样本数: {len(texts)}")
        
        return texts, labels, metadata
    
    def load_combined_dataset(self, max_basic_rumors=5000, max_ced_samples=1000):
        """
        加载合并的数据集
        
        Args:
            max_basic_rumors: 基础谣言数据集最大样本数
            max_ced_samples: CED数据集每类最大样本数
        
        Returns:
            texts: 文本列表
            labels: 标签列表
            metadata: 元数据列表
        """
        all_texts = []
        all_labels = []
        all_metadata = []
        
        # 加载基础谣言数据
        basic_texts, basic_labels, basic_metadata = self.load_basic_rumors(max_basic_rumors)
        all_texts.extend(basic_texts)
        all_labels.extend(basic_labels)
        all_metadata.extend([{**meta, 'source': 'basic'} for meta in basic_metadata])
        
        # 加载CED数据集
        ced_texts, ced_labels, ced_metadata = self.load_ced_dataset(max_ced_samples)
        all_texts.extend(ced_texts)
        all_labels.extend(ced_labels)
        all_metadata.extend([{**meta, 'source': 'ced'} for meta in ced_metadata])
        
        print(f"\n合并数据集统计:")
        print(f"- 总样本数: {len(all_texts)}")
        print(f"- 谣言样本: {sum(all_labels)}")
        print(f"- 非谣言样本: {len(all_labels) - sum(all_labels)}")
        
        return all_texts, all_labels, all_metadata

class DataPreprocessor:
    """数据预处理器"""
    
    @staticmethod
    def clean_text(text):
        """
        清理文本
        
        Args:
            text: 原始文本
        
        Returns:
            cleaned_text: 清理后的文本
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 移除@用户名
        text = re.sub(r'@[^\s]+', '', text)
        
        # 移除话题标签
        text = re.sub(r'#[^#]+#', '', text)
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除emoji（可选）
        # text = re.sub(r'[^\u4e00-\u9fff\w\s\u3000-\u303f\uff00-\uffef]', '', text)
        
        return text.strip()
    
    @staticmethod
    def filter_short_texts(texts, labels, metadata, min_length=10):
        """
        过滤过短的文本
        
        Args:
            texts: 文本列表
            labels: 标签列表
            metadata: 元数据列表
            min_length: 最小文本长度
        
        Returns:
            filtered_texts, filtered_labels, filtered_metadata
        """
        filtered_texts = []
        filtered_labels = []
        filtered_metadata = []
        
        for text, label, meta in zip(texts, labels, metadata):
            if len(text) >= min_length:
                filtered_texts.append(text)
                filtered_labels.append(label)
                filtered_metadata.append(meta)
        
        print(f"过滤后保留 {len(filtered_texts)} 个样本（原始: {len(texts)}）")
        return filtered_texts, filtered_labels, filtered_metadata
    
    @staticmethod
    def balance_dataset(texts, labels, metadata, method='undersample'):
        """
        平衡数据集
        
        Args:
            texts, labels, metadata: 数据
            method: 平衡方法 ('undersample', 'oversample')
        
        Returns:
            balanced_texts, balanced_labels, balanced_metadata
        """
        # 统计各类样本数
        label_counts = Counter(labels)
        print(f"原始数据分布: {dict(label_counts)}")
        
        if len(label_counts) < 2:
            print("数据集只有一个类别，无需平衡")
            return texts, labels, metadata
        
        min_count = min(label_counts.values())
        
        if method == 'undersample':
            # 下采样到最小类别数量
            balanced_texts = []
            balanced_labels = []
            balanced_metadata = []
            
            class_counts = {label: 0 for label in label_counts.keys()}
            
            for text, label, meta in zip(texts, labels, metadata):
                if class_counts[label] < min_count:
                    balanced_texts.append(text)
                    balanced_labels.append(label)
                    balanced_metadata.append(meta)
                    class_counts[label] += 1
            
            print(f"下采样后数据分布: {Counter(balanced_labels)}")
            return balanced_texts, balanced_labels, balanced_metadata
        
        else:
            print("上采样方法暂未实现")
            return texts, labels, metadata

def load_and_preprocess_data(data_dir="/workspace", max_samples=None, 
                           balance=True, min_text_length=10):
    """
    便捷函数：加载和预处理数据
    
    Args:
        data_dir: 数据目录
        max_samples: 最大样本数
        balance: 是否平衡数据集
        min_text_length: 最小文本长度
    
    Returns:
        texts: 预处理后的文本列表
        labels: 标签列表
        metadata: 元数据列表
        data_info: 数据集信息
    """
    # 加载数据
    loader = RumorDataLoader(data_dir)
    
    # 尝试加载CED数据集（包含正负样本）
    texts, labels, metadata = loader.load_ced_dataset(max_samples)
    
    # 如果CED数据集为空，加载基础谣言数据集
    if not texts:
        print("CED数据集为空，尝试加载基础谣言数据集...")
        texts, labels, metadata = loader.load_basic_rumors(max_samples)
        
        # 为基础数据集添加一些非谣言样本（模拟）
        if texts:
            print("注意：基础数据集只包含谣言，正在添加模拟的非谣言样本...")
            non_rumor_texts = [
                "今天天气很好，适合出门运动。",
                "刚刚吃了一顿美味的晚餐，心情很棒。",
                "看了一部有趣的电影，推荐给大家。",
                "工作进展顺利，感谢团队的支持。",
                "周末计划和朋友聚会，很期待。"
            ] * (len(texts) // 10)  # 添加10%的非谣言样本
            
            texts.extend(non_rumor_texts[:len(texts)//2])  # 添加50%的非谣言
            labels.extend([0] * (len(texts) - len(labels)))
            metadata.extend([{'source': 'simulated', 'type': 'non_rumor'}] * (len(texts) - len(metadata)))
    
    # 数据预处理
    preprocessor = DataPreprocessor()
    
    # 清理文本
    texts = [preprocessor.clean_text(text) for text in texts]
    
    # 过滤短文本
    texts, labels, metadata = preprocessor.filter_short_texts(
        texts, labels, metadata, min_text_length
    )
    
    # 平衡数据集
    if balance:
        texts, labels, metadata = preprocessor.balance_dataset(
            texts, labels, metadata, method='undersample'
        )
    
    # 数据集信息
    data_info = {
        'total_samples': len(texts),
        'rumor_samples': sum(labels),
        'non_rumor_samples': len(labels) - sum(labels),
        'avg_text_length': np.mean([len(text) for text in texts]),
        'label_distribution': dict(Counter(labels))
    }
    
    print(f"\n数据预处理完成:")
    print(f"- 总样本数: {data_info['total_samples']}")
    print(f"- 谣言样本: {data_info['rumor_samples']}")
    print(f"- 非谣言样本: {data_info['non_rumor_samples']}")
    print(f"- 平均文本长度: {data_info['avg_text_length']:.1f}")
    
    return texts, labels, metadata, data_info

def split_dataset(texts, labels, metadata, test_size=0.2, random_state=42):
    """
    划分训练集和测试集
    
    Args:
        texts, labels, metadata: 数据
        test_size: 测试集比例
        random_state: 随机种子
    
    Returns:
        train_texts, test_texts, train_labels, test_labels, train_metadata, test_metadata
    """
    from sklearn.model_selection import train_test_split
    
    # 确保数据类型正确
    texts = list(texts)
    labels = list(labels)
    metadata = list(metadata)
    
    # 划分数据
    train_texts, test_texts, train_labels, test_labels, train_metadata, test_metadata = train_test_split(
        texts, labels, metadata, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels  # 保持类别比例
    )
    
    print(f"数据集划分完成:")
    print(f"- 训练集: {len(train_texts)} 样本")
    print(f"- 测试集: {len(test_texts)} 样本")
    print(f"- 训练集谣言比例: {sum(train_labels)/len(train_labels):.3f}")
    print(f"- 测试集谣言比例: {sum(test_labels)/len(test_labels):.3f}")
    
    return train_texts, test_texts, train_labels, test_labels, train_metadata, test_metadata

# 使用示例
if __name__ == "__main__":
    # 加载和预处理数据
    texts, labels, metadata, data_info = load_and_preprocess_data(
        max_samples=1000,  # 限制样本数以便快速测试
        balance=True
    )
    
    # 划分数据集
    train_texts, test_texts, train_labels, test_labels, train_meta, test_meta = split_dataset(
        texts, labels, metadata
    )
    
    # 显示一些示例
    print(f"\n数据示例:")
    for i in range(min(3, len(texts))):
        print(f"文本 {i+1} (标签={labels[i]}): {texts[i][:100]}...")