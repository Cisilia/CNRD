#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文谣言数据集传播特征构建器
Build propagation features from Chinese_Rumor_Dataset

Features to extract:
- 转发数 (Forward count)
- 点赞数 (Like count) 
- 回复数 (Reply count)
- 传播深度 (Propagation depth)
- 级联大小 (Cascade size)
- 距离根节点时间 (Time distance from root node)
- 病毒性分数 (Virality score)
"""

import json
import os
import pandas as pd
from datetime import datetime
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class PropagationFeatureBuilder:
    def __init__(self, dataset_path: str = "/workspace/CED_Dataset"):
        """
        初始化传播特征构建器
        
        Args:
            dataset_path: 数据集路径
        """
        self.dataset_path = dataset_path
        self.original_path = os.path.join(dataset_path, "original-microblog")
        self.rumor_repost_path = os.path.join(dataset_path, "rumor-repost")
        self.non_rumor_repost_path = os.path.join(dataset_path, "non-rumor-repost")
        
        self.features_data = []
        
    def parse_time(self, time_str: str) -> datetime:
        """解析时间字符串"""
        try:
            # 处理原微博时间格式: "Wed Sep 12 00:06:38 +0800 2012"
            if "+" in time_str and len(time_str.split()) == 6:
                # 移除时区信息并重新格式化
                parts = time_str.split()
                time_without_tz = f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[5]}"
                return datetime.strptime(time_without_tz, "%a %b %d %H:%M:%S %Y")
            # 处理转发/评论时间格式: "2012-09-12 23:09:35"
            elif "-" in time_str:
                return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            else:
                # 尝试其他可能的格式
                return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except Exception as e:
            print(f"时间解析错误: {time_str}, 错误: {e}")
            return datetime.now()
    
    def build_cascade_tree(self, reposts: List[Dict]) -> Dict:
        """
        构建级联传播树
        
        Args:
            reposts: 转发/评论数据列表
            
        Returns:
            包含树结构和统计信息的字典
        """
        # 构建节点映射
        nodes = {}
        children = defaultdict(list)
        root_nodes = []
        
        for repost in reposts:
            mid = repost['mid']
            parent = repost.get('parent', '')
            
            nodes[mid] = repost
            
            if parent == '' or parent is None:
                root_nodes.append(mid)
            else:
                children[parent].append(mid)
        
        # 计算传播深度和级联大小
        def calculate_depth_and_size(node_id: str, depth: int = 0) -> Tuple[int, int]:
            """递归计算深度和子树大小"""
            max_depth = depth
            size = 1  # 当前节点
            
            for child_id in children[node_id]:
                child_depth, child_size = calculate_depth_and_size(child_id, depth + 1)
                max_depth = max(max_depth, child_depth)
                size += child_size
                
            return max_depth, size
        
        # 计算每个节点到根节点的距离
        def calculate_distances_from_root(root_time: datetime) -> Dict[str, float]:
            """计算每个节点到根节点的时间距离（小时）"""
            distances = {}
            
            def dfs(node_id: str):
                if node_id in nodes:
                    node_time = self.parse_time(nodes[node_id]['date'])
                    time_diff = (node_time - root_time).total_seconds() / 3600  # 转换为小时
                    distances[node_id] = max(0, time_diff)  # 确保非负
                    
                    for child_id in children[node_id]:
                        dfs(child_id)
            
            for root_id in root_nodes:
                dfs(root_id)
                
            return distances
        
        # 获取根节点时间（如果有多个根节点，取最早的）
        root_time = None
        if root_nodes:
            root_times = [self.parse_time(nodes[root_id]['date']) for root_id in root_nodes if root_id in nodes]
            if root_times:
                root_time = min(root_times)
        
        # 计算整体统计
        total_depth = 0
        total_size = len(reposts)
        
        if root_nodes:
            depths_and_sizes = [calculate_depth_and_size(root_id) for root_id in root_nodes if root_id in nodes]
            if depths_and_sizes:
                total_depth = max(depth for depth, _ in depths_and_sizes)
        
        # 计算时间距离
        time_distances = {}
        if root_time:
            time_distances = calculate_distances_from_root(root_time)
        
        return {
            'total_depth': total_depth,
            'cascade_size': total_size,
            'root_nodes': root_nodes,
            'children_map': dict(children),
            'time_distances': time_distances,
            'nodes': nodes
        }
    
    def calculate_virality_score(self, forward_count: int, like_count: int, 
                               reply_count: int, cascade_size: int, 
                               max_depth: int, avg_time_to_spread: float) -> float:
        """
        计算病毒性分数
        
        综合考虑多个传播指标的加权得分
        """
        # 标准化各个指标 (使用对数变换来处理偏斜分布)
        log_forward = np.log1p(forward_count)
        log_like = np.log1p(like_count) 
        log_reply = np.log1p(reply_count)
        log_cascade = np.log1p(cascade_size)
        log_depth = np.log1p(max_depth)
        
        # 时间传播速度 (越快传播病毒性越高)
        time_factor = 1.0 / (1.0 + avg_time_to_spread / 24.0) if avg_time_to_spread > 0 else 1.0
        
        # 加权计算病毒性分数
        virality = (
            0.25 * log_forward +      # 转发权重
            0.15 * log_like +         # 点赞权重  
            0.20 * log_reply +        # 回复权重
            0.20 * log_cascade +      # 级联大小权重
            0.10 * log_depth +        # 传播深度权重
            0.10 * time_factor        # 时间因子权重
        )
        
        return round(virality, 4)
    
    def process_microblog(self, original_file: str, repost_file: str, is_rumor: bool) -> Dict:
        """
        处理单个微博及其传播数据
        
        Args:
            original_file: 原微博文件路径
            repost_file: 转发/评论文件路径
            is_rumor: 是否为谣言
            
        Returns:
            包含所有传播特征的字典
        """
        features = {}
        
        # 读取原微博数据
        try:
            with open(original_file, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
        except Exception as e:
            print(f"读取原微博文件失败: {original_file}, 错误: {e}")
            return None
            
        # 提取基本信息
        features['microblog_id'] = os.path.basename(original_file).replace('.json', '')
        features['is_rumor'] = is_rumor
        features['original_text'] = original_data.get('text', '')
        features['original_time'] = original_data.get('time', '')
        
        # 从原微博数据中获取基础统计
        features['like_count'] = original_data.get('likes', 0)
        features['original_forward_count'] = original_data.get('reposts', 0)
        features['original_reply_count'] = original_data.get('comments', 0)
        
        # 读取转发/评论数据
        reposts = []
        if os.path.exists(repost_file):
            try:
                with open(repost_file, 'r', encoding='utf-8') as f:
                    reposts = json.load(f)
            except Exception as e:
                print(f"读取转发文件失败: {repost_file}, 错误: {e}")
                reposts = []
        
        # 计算实际传播统计
        features['actual_cascade_size'] = len(reposts)
        
        # 区分转发和回复 (有文本内容的为回复，无文本的为纯转发)
        forwards = [r for r in reposts if r.get('text', '').strip() == '']
        replies = [r for r in reposts if r.get('text', '').strip() != '']
        
        features['forward_count'] = len(forwards)
        features['reply_count'] = len(replies)
        
        # 构建级联树并计算传播特征
        if reposts:
            cascade_info = self.build_cascade_tree(reposts)
            features['propagation_depth'] = cascade_info['total_depth']
            features['cascade_size'] = cascade_info['cascade_size']
            
            # 计算时间相关特征
            if cascade_info['time_distances']:
                time_distances = list(cascade_info['time_distances'].values())
                features['avg_time_from_root'] = np.mean(time_distances)
                features['max_time_from_root'] = np.max(time_distances)
                features['min_time_from_root'] = np.min(time_distances)
                features['std_time_from_root'] = np.std(time_distances)
            else:
                features['avg_time_from_root'] = 0.0
                features['max_time_from_root'] = 0.0
                features['min_time_from_root'] = 0.0
                features['std_time_from_root'] = 0.0
        else:
            features['propagation_depth'] = 0
            features['cascade_size'] = 0
            features['avg_time_from_root'] = 0.0
            features['max_time_from_root'] = 0.0
            features['min_time_from_root'] = 0.0
            features['std_time_from_root'] = 0.0
        
        # 计算病毒性分数
        features['virality_score'] = self.calculate_virality_score(
            features['forward_count'],
            features['like_count'], 
            features['reply_count'],
            features['cascade_size'],
            features['propagation_depth'],
            features['avg_time_from_root']
        )
        
        return features
    
    def build_features(self) -> pd.DataFrame:
        """
        构建所有微博的传播特征
        
        Returns:
            包含所有传播特征的DataFrame
        """
        print("开始构建传播特征...")
        
        # 获取所有原微博文件
        original_files = []
        for file in os.listdir(self.original_path):
            if file.endswith('.json'):
                original_files.append(file)
        
        print(f"找到 {len(original_files)} 个原微博文件")
        
        # 处理每个微博
        processed_count = 0
        for original_file in original_files:
            original_file_path = os.path.join(self.original_path, original_file)
            
            # 确定是谣言还是非谣言
            rumor_repost_file = os.path.join(self.rumor_repost_path, original_file)
            non_rumor_repost_file = os.path.join(self.non_rumor_repost_path, original_file)
            
            if os.path.exists(rumor_repost_file):
                # 谣言
                features = self.process_microblog(original_file_path, rumor_repost_file, True)
            elif os.path.exists(non_rumor_repost_file):
                # 非谣言
                features = self.process_microblog(original_file_path, non_rumor_repost_file, False)
            else:
                # 没有传播数据的微博
                features = self.process_microblog(original_file_path, "", False)
            
            if features:
                self.features_data.append(features)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"已处理 {processed_count} 个微博...")
        
        print(f"总共处理了 {processed_count} 个微博")
        
        # 转换为DataFrame
        df = pd.DataFrame(self.features_data)
        return df
    
    def add_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加额外的传播特征
        
        Args:
            df: 基础特征DataFrame
            
        Returns:
            增强后的DataFrame
        """
        print("添加额外的传播特征...")
        
        # 传播效率 = 级联大小 / 传播深度
        df['propagation_efficiency'] = df.apply(
            lambda row: row['cascade_size'] / max(row['propagation_depth'], 1), axis=1
        )
        
        # 互动率 = (转发数 + 回复数) / 级联大小
        df['interaction_rate'] = df.apply(
            lambda row: (row['forward_count'] + row['reply_count']) / max(row['cascade_size'], 1), axis=1
        )
        
        # 点赞转发比
        df['like_forward_ratio'] = df.apply(
            lambda row: row['like_count'] / max(row['forward_count'], 1), axis=1
        )
        
        # 回复转发比
        df['reply_forward_ratio'] = df.apply(
            lambda row: row['reply_count'] / max(row['forward_count'], 1), axis=1
        )
        
        # 传播速度 = 级联大小 / 平均传播时间
        df['propagation_speed'] = df.apply(
            lambda row: row['cascade_size'] / max(row['avg_time_from_root'], 0.1), axis=1
        )
        
        # 传播广度 (根据标准差衡量传播时间的分散程度)
        df['propagation_breadth'] = df['std_time_from_root']
        
        return df
    
    def generate_statistics(self, df: pd.DataFrame) -> Dict:
        """
        生成数据集统计信息
        
        Args:
            df: 特征DataFrame
            
        Returns:
            统计信息字典
        """
        stats = {}
        
        # 基本统计
        stats['total_microblogs'] = len(df)
        stats['rumor_count'] = len(df[df['is_rumor'] == True])
        stats['non_rumor_count'] = len(df[df['is_rumor'] == False])
        
        # 传播特征统计
        numeric_features = [
            'forward_count', 'like_count', 'reply_count', 
            'propagation_depth', 'cascade_size', 'avg_time_from_root',
            'virality_score', 'propagation_efficiency', 'interaction_rate'
        ]
        
        for feature in numeric_features:
            if feature in df.columns:
                stats[f'{feature}_mean'] = df[feature].mean()
                stats[f'{feature}_std'] = df[feature].std()
                stats[f'{feature}_median'] = df[feature].median()
                stats[f'{feature}_max'] = df[feature].max()
                stats[f'{feature}_min'] = df[feature].min()
        
        # 谣言vs非谣言对比
        rumor_df = df[df['is_rumor'] == True]
        non_rumor_df = df[df['is_rumor'] == False]
        
        stats['rumor_vs_non_rumor'] = {}
        for feature in numeric_features:
            if feature in df.columns:
                stats['rumor_vs_non_rumor'][feature] = {
                    'rumor_mean': rumor_df[feature].mean() if len(rumor_df) > 0 else 0,
                    'non_rumor_mean': non_rumor_df[feature].mean() if len(non_rumor_df) > 0 else 0,
                    'difference': (rumor_df[feature].mean() - non_rumor_df[feature].mean()) if len(rumor_df) > 0 and len(non_rumor_df) > 0 else 0
                }
        
        return stats
    
    def save_results(self, df: pd.DataFrame, stats: Dict, output_dir: str = "/workspace/results"):
        """
        保存结果到文件
        
        Args:
            df: 特征DataFrame
            stats: 统计信息
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存特征数据
        df.to_csv(os.path.join(output_dir, 'propagation_features.csv'), index=False, encoding='utf-8')
        df.to_json(os.path.join(output_dir, 'propagation_features.json'), orient='records', ensure_ascii=False)
        
        # 保存统计信息
        with open(os.path.join(output_dir, 'statistics.json'), 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"结果已保存到 {output_dir}")
        print(f"- 特征数据: propagation_features.csv, propagation_features.json")
        print(f"- 统计信息: statistics.json")
    
    def create_visualizations(self, df: pd.DataFrame, output_dir: str = "/workspace/results"):
        """
        创建可视化图表
        
        Args:
            df: 特征DataFrame
            output_dir: 输出目录
        """
        print("生成可视化图表...")
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图表目录
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. 谣言vs非谣言传播特征对比
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('谣言vs非谣言传播特征对比', fontsize=16)
        
        features_to_plot = [
            ('forward_count', '转发数'),
            ('like_count', '点赞数'), 
            ('reply_count', '回复数'),
            ('propagation_depth', '传播深度'),
            ('cascade_size', '级联大小'),
            ('virality_score', '病毒性分数')
        ]
        
        for i, (feature, title) in enumerate(features_to_plot):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            rumor_data = df[df['is_rumor'] == True][feature]
            non_rumor_data = df[df['is_rumor'] == False][feature]
            
            ax.hist([rumor_data, non_rumor_data], bins=30, alpha=0.7, 
                   label=['谣言', '非谣言'], color=['red', 'blue'])
            ax.set_title(title)
            ax.set_xlabel(title)
            ax.set_ylabel('频次')
            ax.legend()
            ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'rumor_vs_non_rumor_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 传播特征相关性热力图
        plt.figure(figsize=(12, 10))
        numeric_features = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_features].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('传播特征相关性热力图')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'feature_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 病毒性分数分布
        plt.figure(figsize=(10, 6))
        plt.hist(df[df['is_rumor'] == True]['virality_score'], bins=50, alpha=0.7, 
                label='谣言', color='red', density=True)
        plt.hist(df[df['is_rumor'] == False]['virality_score'], bins=50, alpha=0.7,
                label='非谣言', color='blue', density=True)
        plt.xlabel('病毒性分数')
        plt.ylabel('密度')
        plt.title('病毒性分数分布对比')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(viz_dir, 'virality_score_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可视化图表已保存到 {viz_dir}")
    
    def run(self, output_dir: str = "/workspace/results"):
        """
        运行完整的特征构建流程
        
        Args:
            output_dir: 输出目录
        """
        print("=== 中文谣言数据集传播特征构建 ===")
        
        # 构建特征
        df = self.build_features()
        
        if df.empty:
            print("没有成功处理任何数据")
            return
        
        # 添加额外特征
        df = self.add_additional_features(df)
        
        # 生成统计信息
        stats = self.generate_statistics(df)
        
        # 保存结果
        self.save_results(df, stats, output_dir)
        
        # 创建可视化
        self.create_visualizations(df, output_dir)
        
        # 打印摘要
        print("\n=== 特征构建完成 ===")
        print(f"总微博数: {len(df)}")
        print(f"谣言数: {stats['rumor_count']}")
        print(f"非谣言数: {stats['non_rumor_count']}")
        print(f"平均传播深度: {stats.get('propagation_depth_mean', 0):.2f}")
        print(f"平均级联大小: {stats.get('cascade_size_mean', 0):.2f}")
        print(f"平均病毒性分数: {stats.get('virality_score_mean', 0):.4f}")
        
        return df

def main():
    """主函数"""
    builder = PropagationFeatureBuilder()
    df = builder.run()
    
    # 显示前几行数据
    print("\n=== 特征数据预览 ===")
    print(df.head())
    
    print("\n=== 特征列表 ===")
    for col in df.columns:
        print(f"- {col}")

if __name__ == "__main__":
    main()