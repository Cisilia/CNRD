#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文谣言数据集传播特征构建器 (简化版)
Build propagation features from Chinese_Rumor_Dataset using only built-in libraries

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
import csv
from datetime import datetime
import math
from collections import defaultdict
import statistics

class PropagationFeatureBuilder:
    def __init__(self, dataset_path="/workspace/CED_Dataset"):
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
        
    def parse_time(self, time_str):
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
    
    def build_cascade_tree(self, reposts):
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
        def calculate_depth_and_size(node_id, depth=0):
            """递归计算深度和子树大小"""
            max_depth = depth
            size = 1  # 当前节点
            
            for child_id in children[node_id]:
                child_depth, child_size = calculate_depth_and_size(child_id, depth + 1)
                max_depth = max(max_depth, child_depth)
                size += child_size
                
            return max_depth, size
        
        # 计算每个节点到根节点的距离
        def calculate_distances_from_root(root_time):
            """计算每个节点到根节点的时间距离（小时）"""
            distances = {}
            
            def dfs(node_id):
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
    
    def calculate_virality_score(self, forward_count, like_count, reply_count, 
                               cascade_size, max_depth, avg_time_to_spread):
        """
        计算病毒性分数
        
        综合考虑多个传播指标的加权得分
        """
        # 标准化各个指标 (使用对数变换来处理偏斜分布)
        log_forward = math.log1p(forward_count)
        log_like = math.log1p(like_count) 
        log_reply = math.log1p(reply_count)
        log_cascade = math.log1p(cascade_size)
        log_depth = math.log1p(max_depth)
        
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
    
    def process_microblog(self, original_file, repost_file, is_rumor):
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
        if repost_file and os.path.exists(repost_file):
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
                features['avg_time_from_root'] = statistics.mean(time_distances)
                features['max_time_from_root'] = max(time_distances)
                features['min_time_from_root'] = min(time_distances)
                features['std_time_from_root'] = statistics.stdev(time_distances) if len(time_distances) > 1 else 0.0
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
        
        # 添加额外的传播特征
        # 传播效率 = 级联大小 / 传播深度
        features['propagation_efficiency'] = features['cascade_size'] / max(features['propagation_depth'], 1)
        
        # 互动率 = (转发数 + 回复数) / 级联大小
        features['interaction_rate'] = (features['forward_count'] + features['reply_count']) / max(features['cascade_size'], 1)
        
        # 点赞转发比
        features['like_forward_ratio'] = features['like_count'] / max(features['forward_count'], 1)
        
        # 回复转发比
        features['reply_forward_ratio'] = features['reply_count'] / max(features['forward_count'], 1)
        
        # 传播速度 = 级联大小 / 平均传播时间
        features['propagation_speed'] = features['cascade_size'] / max(features['avg_time_from_root'], 0.1)
        
        # 传播广度 (根据标准差衡量传播时间的分散程度)
        features['propagation_breadth'] = features['std_time_from_root']
        
        return features
    
    def build_features(self):
        """
        构建所有微博的传播特征
        
        Returns:
            包含所有传播特征的列表
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
        return self.features_data
    
    def generate_statistics(self, features_data):
        """
        生成数据集统计信息
        
        Args:
            features_data: 特征数据列表
            
        Returns:
            统计信息字典
        """
        stats = {}
        
        # 基本统计
        stats['total_microblogs'] = len(features_data)
        rumor_data = [f for f in features_data if f['is_rumor']]
        non_rumor_data = [f for f in features_data if not f['is_rumor']]
        
        stats['rumor_count'] = len(rumor_data)
        stats['non_rumor_count'] = len(non_rumor_data)
        
        # 传播特征统计
        numeric_features = [
            'forward_count', 'like_count', 'reply_count', 
            'propagation_depth', 'cascade_size', 'avg_time_from_root',
            'virality_score', 'propagation_efficiency', 'interaction_rate'
        ]
        
        for feature in numeric_features:
            values = [f[feature] for f in features_data if feature in f]
            if values:
                stats[f'{feature}_mean'] = statistics.mean(values)
                stats[f'{feature}_median'] = statistics.median(values)
                stats[f'{feature}_max'] = max(values)
                stats[f'{feature}_min'] = min(values)
                if len(values) > 1:
                    stats[f'{feature}_std'] = statistics.stdev(values)
                else:
                    stats[f'{feature}_std'] = 0.0
        
        # 谣言vs非谣言对比
        stats['rumor_vs_non_rumor'] = {}
        for feature in numeric_features:
            rumor_values = [f[feature] for f in rumor_data if feature in f]
            non_rumor_values = [f[feature] for f in non_rumor_data if feature in f]
            
            rumor_mean = statistics.mean(rumor_values) if rumor_values else 0
            non_rumor_mean = statistics.mean(non_rumor_values) if non_rumor_values else 0
            
            stats['rumor_vs_non_rumor'][feature] = {
                'rumor_mean': rumor_mean,
                'non_rumor_mean': non_rumor_mean,
                'difference': rumor_mean - non_rumor_mean
            }
        
        return stats
    
    def save_results(self, features_data, stats, output_dir="/workspace/results"):
        """
        保存结果到文件
        
        Args:
            features_data: 特征数据列表
            stats: 统计信息
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存特征数据为CSV
        if features_data:
            # 获取所有可能的列名
            all_columns = set()
            for feature in features_data:
                all_columns.update(feature.keys())
            all_columns = sorted(list(all_columns))
            
            # 写入CSV文件
            csv_file = os.path.join(output_dir, 'propagation_features.csv')
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_columns)
                writer.writeheader()
                writer.writerows(features_data)
        
        # 保存特征数据为JSON
        json_file = os.path.join(output_dir, 'propagation_features.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(features_data, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存统计信息
        stats_file = os.path.join(output_dir, 'statistics.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"结果已保存到 {output_dir}")
        print(f"- 特征数据: propagation_features.csv, propagation_features.json")
        print(f"- 统计信息: statistics.json")
    
    def run(self, output_dir="/workspace/results"):
        """
        运行完整的特征构建流程
        
        Args:
            output_dir: 输出目录
        """
        print("=== 中文谣言数据集传播特征构建 ===")
        
        # 构建特征
        features_data = self.build_features()
        
        if not features_data:
            print("没有成功处理任何数据")
            return
        
        # 生成统计信息
        stats = self.generate_statistics(features_data)
        
        # 保存结果
        self.save_results(features_data, stats, output_dir)
        
        # 打印摘要
        print("\n=== 特征构建完成 ===")
        print(f"总微博数: {len(features_data)}")
        print(f"谣言数: {stats['rumor_count']}")
        print(f"非谣言数: {stats['non_rumor_count']}")
        print(f"平均传播深度: {stats.get('propagation_depth_mean', 0):.2f}")
        print(f"平均级联大小: {stats.get('cascade_size_mean', 0):.2f}")
        print(f"平均病毒性分数: {stats.get('virality_score_mean', 0):.4f}")
        
        # 显示前几行数据
        print("\n=== 特征数据预览 ===")
        if features_data:
            print("特征列表:")
            for key in sorted(features_data[0].keys()):
                print(f"- {key}")
            
            print("\n前3个样本:")
            for i, sample in enumerate(features_data[:3]):
                print(f"\n样本 {i+1}:")
                for key, value in sample.items():
                    if key != 'original_text':  # 跳过长文本
                        print(f"  {key}: {value}")
        
        return features_data

def main():
    """主函数"""
    builder = PropagationFeatureBuilder()
    features_data = builder.run()
    
    print(f"\n构建完成！共生成 {len(features_data)} 条记录的传播特征")

if __name__ == "__main__":
    main()