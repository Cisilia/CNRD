#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文谣言数据集传播特征构建器 (增强版)
Enhanced propagation features builder with improved time parsing

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
import re

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
        
        # 中文月份映射
        self.chinese_months = {
            '01月': '01', '02月': '02', '03月': '03', '04月': '04',
            '05月': '05', '06月': '06', '07月': '07', '08月': '08',
            '09月': '09', '10月': '10', '11月': '11', '12月': '12'
        }
        
    def parse_time(self, time_str):
        """解析时间字符串，支持多种格式"""
        try:
            # 处理时间戳格式
            if isinstance(time_str, (int, float)) or (isinstance(time_str, str) and time_str.isdigit()):
                return datetime.fromtimestamp(int(time_str))
            
            # 处理原微博时间格式: "Wed Sep 12 00:06:38 +0800 2012"
            if "+" in time_str and len(time_str.split()) == 6:
                # 移除时区信息并重新格式化
                parts = time_str.split()
                time_without_tz = f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[5]}"
                return datetime.strptime(time_without_tz, "%a %b %d %H:%M:%S %Y")
            
            # 处理转发/评论时间格式: "2012-09-12 23:09:35"
            elif "-" in time_str and ":" in time_str:
                return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            
            # 处理中文日期格式: "02月23日 09:49"
            elif "月" in time_str and "日" in time_str:
                # 使用正则表达式提取月、日、时、分
                pattern = r'(\d{1,2})月(\d{1,2})日\s+(\d{1,2}):(\d{1,2})'
                match = re.match(pattern, time_str)
                if match:
                    month, day, hour, minute = match.groups()
                    # 假设年份为2012年（数据集中的常见年份）
                    year = 2012
                    return datetime(year, int(month), int(day), int(hour), int(minute))
            
            # 其他格式尝试
            else:
                return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                
        except Exception as e:
            # 如果所有格式都失败，返回默认时间
            print(f"时间解析失败: {time_str}, 错误: {e}")
            return datetime(2012, 1, 1)  # 返回默认时间
    
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
        
        # 计算传播深度
        def calculate_max_depth(node_id, depth=0):
            """递归计算最大深度"""
            max_depth = depth
            for child_id in children[node_id]:
                child_depth = calculate_max_depth(child_id, depth + 1)
                max_depth = max(max_depth, child_depth)
            return max_depth
        
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
            root_times = []
            for root_id in root_nodes:
                if root_id in nodes:
                    try:
                        root_times.append(self.parse_time(nodes[root_id]['date']))
                    except:
                        continue
            if root_times:
                root_time = min(root_times)
        
        # 计算整体统计
        total_depth = 0
        total_size = len(reposts)
        
        if root_nodes:
            depths = [calculate_max_depth(root_id) for root_id in root_nodes if root_id in nodes]
            if depths:
                total_depth = max(depths)
        
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
    
    def extract_text_features(self, text):
        """
        提取文本特征
        
        Args:
            text: 微博文本
            
        Returns:
            文本特征字典
        """
        features = {}
        
        # 文本长度
        features['text_length'] = len(text)
        
        # 是否包含URL
        features['has_url'] = 1 if ('http' in text or 'www.' in text) else 0
        
        # 是否包含@提及
        features['has_mention'] = 1 if '@' in text else 0
        
        # 是否包含话题标签
        features['has_hashtag'] = 1 if '#' in text else 0
        
        # 感叹号数量
        features['exclamation_count'] = text.count('！') + text.count('!')
        
        # 问号数量
        features['question_count'] = text.count('？') + text.count('?')
        
        return features
    
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
        
        # 提取文本特征
        text_features = self.extract_text_features(features['original_text'])
        features.update(text_features)
        
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
                features['median_time_from_root'] = statistics.median(time_distances)
            else:
                features['avg_time_from_root'] = 0.0
                features['max_time_from_root'] = 0.0
                features['min_time_from_root'] = 0.0
                features['std_time_from_root'] = 0.0
                features['median_time_from_root'] = 0.0
        else:
            features['propagation_depth'] = 0
            features['cascade_size'] = 0
            features['avg_time_from_root'] = 0.0
            features['max_time_from_root'] = 0.0
            features['min_time_from_root'] = 0.0
            features['std_time_from_root'] = 0.0
            features['median_time_from_root'] = 0.0
        
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
        
        # 传播持续时间
        features['propagation_duration'] = features['max_time_from_root'] - features['min_time_from_root']
        
        # 早期传播强度 (前6小时的传播量)
        early_reposts = 0
        if reposts:
            for repost in reposts:
                try:
                    repost_time = self.parse_time(repost['date'])
                    original_time = self.parse_time(features['original_time'])
                    hours_diff = (repost_time - original_time).total_seconds() / 3600
                    if 0 <= hours_diff <= 6:
                        early_reposts += 1
                except:
                    continue
        features['early_propagation_intensity'] = early_reposts
        
        # 传播活跃度 = 级联大小 / 传播持续时间
        features['propagation_activity'] = features['cascade_size'] / max(features['propagation_duration'], 1)
        
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
        error_count = 0
        
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
            else:
                error_count += 1
                
            if processed_count % 500 == 0:
                print(f"已处理 {processed_count} 个微博...")
        
        print(f"总共处理了 {processed_count} 个微博，{error_count} 个失败")
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
            'virality_score', 'propagation_efficiency', 'interaction_rate',
            'propagation_speed', 'propagation_duration', 'early_propagation_intensity',
            'text_length', 'has_url', 'has_mention', 'has_hashtag'
        ]
        
        for feature in numeric_features:
            values = [f[feature] for f in features_data if feature in f and f[feature] is not None]
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
            rumor_values = [f[feature] for f in rumor_data if feature in f and f[feature] is not None]
            non_rumor_values = [f[feature] for f in non_rumor_data if feature in f and f[feature] is not None]
            
            rumor_mean = statistics.mean(rumor_values) if rumor_values else 0
            non_rumor_mean = statistics.mean(non_rumor_values) if non_rumor_values else 0
            
            stats['rumor_vs_non_rumor'][feature] = {
                'rumor_mean': rumor_mean,
                'non_rumor_mean': non_rumor_mean,
                'difference': rumor_mean - non_rumor_mean,
                'rumor_median': statistics.median(rumor_values) if rumor_values else 0,
                'non_rumor_median': statistics.median(non_rumor_values) if non_rumor_values else 0
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
            csv_file = os.path.join(output_dir, 'propagation_features_enhanced.csv')
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_columns)
                writer.writeheader()
                writer.writerows(features_data)
        
        # 保存特征数据为JSON
        json_file = os.path.join(output_dir, 'propagation_features_enhanced.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(features_data, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存统计信息
        stats_file = os.path.join(output_dir, 'statistics_enhanced.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        
        # 创建特征说明文档
        feature_desc_file = os.path.join(output_dir, 'feature_descriptions.txt')
        with open(feature_desc_file, 'w', encoding='utf-8') as f:
            f.write("=== 传播特征说明 ===\n\n")
            f.write("基础特征:\n")
            f.write("- microblog_id: 微博唯一标识\n")
            f.write("- is_rumor: 是否为谣言 (True/False)\n")
            f.write("- original_text: 原微博文本内容\n")
            f.write("- original_time: 原微博发布时间\n\n")
            
            f.write("传播统计特征:\n")
            f.write("- forward_count: 实际转发数\n")
            f.write("- like_count: 点赞数\n")
            f.write("- reply_count: 回复数\n")
            f.write("- propagation_depth: 传播深度 (最大层级)\n")
            f.write("- cascade_size: 级联大小 (总传播节点数)\n")
            f.write("- actual_cascade_size: 实际级联大小\n\n")
            
            f.write("时间特征:\n")
            f.write("- avg_time_from_root: 平均距离根节点时间 (小时)\n")
            f.write("- max_time_from_root: 最大距离根节点时间 (小时)\n")
            f.write("- min_time_from_root: 最小距离根节点时间 (小时)\n")
            f.write("- std_time_from_root: 距离根节点时间标准差\n")
            f.write("- median_time_from_root: 距离根节点时间中位数\n")
            f.write("- propagation_duration: 传播持续时间\n")
            f.write("- early_propagation_intensity: 早期传播强度 (前6小时)\n\n")
            
            f.write("计算特征:\n")
            f.write("- virality_score: 病毒性分数\n")
            f.write("- propagation_efficiency: 传播效率\n")
            f.write("- interaction_rate: 互动率\n")
            f.write("- like_forward_ratio: 点赞转发比\n")
            f.write("- reply_forward_ratio: 回复转发比\n")
            f.write("- propagation_speed: 传播速度\n")
            f.write("- propagation_breadth: 传播广度\n")
            f.write("- propagation_activity: 传播活跃度\n\n")
            
            f.write("文本特征:\n")
            f.write("- text_length: 文本长度\n")
            f.write("- has_url: 是否包含URL\n")
            f.write("- has_mention: 是否包含@提及\n")
            f.write("- has_hashtag: 是否包含话题标签\n")
            f.write("- exclamation_count: 感叹号数量\n")
            f.write("- question_count: 问号数量\n")
        
        print(f"结果已保存到 {output_dir}")
        print(f"- 特征数据: propagation_features_enhanced.csv, propagation_features_enhanced.json")
        print(f"- 统计信息: statistics_enhanced.json")
        print(f"- 特征说明: feature_descriptions.txt")
    
    def print_summary(self, features_data, stats):
        """打印特征构建摘要"""
        print("\n=== 特征构建完成 ===")
        print(f"总微博数: {len(features_data)}")
        print(f"谣言数: {stats['rumor_count']}")
        print(f"非谣言数: {stats['non_rumor_count']}")
        
        print(f"\n=== 传播特征统计 ===")
        print(f"平均传播深度: {stats.get('propagation_depth_mean', 0):.2f}")
        print(f"平均级联大小: {stats.get('cascade_size_mean', 0):.2f}")
        print(f"平均病毒性分数: {stats.get('virality_score_mean', 0):.4f}")
        print(f"平均传播速度: {stats.get('propagation_speed_mean', 0):.2f}")
        
        print(f"\n=== 谣言vs非谣言对比 ===")
        key_features = ['virality_score', 'cascade_size', 'propagation_depth', 'forward_count']
        for feature in key_features:
            if feature in stats['rumor_vs_non_rumor']:
                rumor_mean = stats['rumor_vs_non_rumor'][feature]['rumor_mean']
                non_rumor_mean = stats['rumor_vs_non_rumor'][feature]['non_rumor_mean']
                diff = stats['rumor_vs_non_rumor'][feature]['difference']
                print(f"{feature}: 谣言={rumor_mean:.2f}, 非谣言={non_rumor_mean:.2f}, 差异={diff:.2f}")
    
    def run(self, output_dir="/workspace/results"):
        """
        运行完整的特征构建流程
        
        Args:
            output_dir: 输出目录
        """
        print("=== 中文谣言数据集传播特征构建 (增强版) ===")
        
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
        self.print_summary(features_data, stats)
        
        return features_data

def main():
    """主函数"""
    builder = PropagationFeatureBuilder()
    features_data = builder.run()
    
    print(f"\n构建完成！共生成 {len(features_data)} 条记录的传播特征")

if __name__ == "__main__":
    main()