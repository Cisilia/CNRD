#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传播特征分析脚本
Analyze and visualize propagation features from Chinese Rumor Dataset
"""

import json
import csv
from collections import Counter
import statistics
import math

def load_features(file_path="/workspace/results/propagation_features_enhanced.csv"):
    """加载传播特征数据"""
    features_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 转换数值类型
            for key, value in row.items():
                if key in ['is_rumor']:
                    row[key] = value.lower() == 'true'
                elif key not in ['microblog_id', 'original_text', 'original_time']:
                    try:
                        row[key] = float(value) if '.' in value else int(value)
                    except:
                        pass
            features_data.append(row)
    
    return features_data

def analyze_feature_distributions(features_data):
    """分析特征分布"""
    print("=== 特征分布分析 ===")
    
    # 分离谣言和非谣言数据
    rumor_data = [f for f in features_data if f['is_rumor']]
    non_rumor_data = [f for f in features_data if not f['is_rumor']]
    
    print(f"谣言样本: {len(rumor_data)}")
    print(f"非谣言样本: {len(non_rumor_data)}")
    
    # 关键特征分析
    key_features = [
        'forward_count', 'like_count', 'reply_count', 
        'propagation_depth', 'cascade_size', 'virality_score',
        'propagation_speed', 'early_propagation_intensity'
    ]
    
    print("\n=== 关键特征对比 ===")
    for feature in key_features:
        rumor_values = [f[feature] for f in rumor_data if feature in f]
        non_rumor_values = [f[feature] for f in non_rumor_data if feature in f]
        
        if rumor_values and non_rumor_values:
            print(f"\n{feature}:")
            print(f"  谣言 - 均值: {statistics.mean(rumor_values):.2f}, 中位数: {statistics.median(rumor_values):.2f}")
            print(f"  非谣言 - 均值: {statistics.mean(non_rumor_values):.2f}, 中位数: {statistics.median(non_rumor_values):.2f}")
            print(f"  差异: {statistics.mean(rumor_values) - statistics.mean(non_rumor_values):.2f}")

def analyze_high_virality_content(features_data, top_n=10):
    """分析高病毒性内容"""
    print(f"\n=== 高病毒性内容分析 (Top {top_n}) ===")
    
    # 按病毒性分数排序
    sorted_data = sorted(features_data, key=lambda x: x['virality_score'], reverse=True)
    
    print("病毒性分数最高的内容:")
    for i, item in enumerate(sorted_data[:top_n]):
        print(f"\n{i+1}. 病毒性分数: {item['virality_score']}")
        print(f"   类型: {'谣言' if item['is_rumor'] else '非谣言'}")
        print(f"   转发数: {item['forward_count']}, 回复数: {item['reply_count']}, 点赞数: {item['like_count']}")
        print(f"   传播深度: {item['propagation_depth']}, 级联大小: {item['cascade_size']}")
        print(f"   内容: {item['original_text'][:100]}...")

def analyze_propagation_patterns(features_data):
    """分析传播模式"""
    print("\n=== 传播模式分析 ===")
    
    # 传播深度分析
    depth_counter = Counter([f['propagation_depth'] for f in features_data])
    print("\n传播深度分布:")
    for depth in sorted(depth_counter.keys())[:10]:
        print(f"  深度 {depth}: {depth_counter[depth]} 条")
    
    # 级联大小分析
    cascade_sizes = [f['cascade_size'] for f in features_data]
    print(f"\n级联大小统计:")
    print(f"  最小: {min(cascade_sizes)}")
    print(f"  最大: {max(cascade_sizes)}")
    print(f"  平均: {statistics.mean(cascade_sizes):.2f}")
    print(f"  中位数: {statistics.median(cascade_sizes):.2f}")
    
    # 传播速度分析
    speeds = [f['propagation_speed'] for f in features_data if f['propagation_speed'] > 0]
    if speeds:
        print(f"\n传播速度统计:")
        print(f"  最快: {max(speeds):.2f}")
        print(f"  最慢: {min(speeds):.2f}")
        print(f"  平均: {statistics.mean(speeds):.2f}")

def analyze_text_features(features_data):
    """分析文本特征"""
    print("\n=== 文本特征分析 ===")
    
    # 分离谣言和非谣言
    rumor_data = [f for f in features_data if f['is_rumor']]
    non_rumor_data = [f for f in features_data if not f['is_rumor']]
    
    # URL使用率
    rumor_url_rate = sum(f['has_url'] for f in rumor_data) / len(rumor_data) * 100
    non_rumor_url_rate = sum(f['has_url'] for f in non_rumor_data) / len(non_rumor_data) * 100
    print(f"URL使用率: 谣言 {rumor_url_rate:.1f}%, 非谣言 {non_rumor_url_rate:.1f}%")
    
    # @提及使用率
    rumor_mention_rate = sum(f['has_mention'] for f in rumor_data) / len(rumor_data) * 100
    non_rumor_mention_rate = sum(f['has_mention'] for f in non_rumor_data) / len(non_rumor_data) * 100
    print(f"@提及使用率: 谣言 {rumor_mention_rate:.1f}%, 非谣言 {non_rumor_mention_rate:.1f}%")
    
    # 话题标签使用率
    rumor_hashtag_rate = sum(f['has_hashtag'] for f in rumor_data) / len(rumor_data) * 100
    non_rumor_hashtag_rate = sum(f['has_hashtag'] for f in non_rumor_data) / len(non_rumor_data) * 100
    print(f"话题标签使用率: 谣言 {rumor_hashtag_rate:.1f}%, 非谣言 {non_rumor_hashtag_rate:.1f}%")
    
    # 文本长度对比
    rumor_text_lengths = [f['text_length'] for f in rumor_data]
    non_rumor_text_lengths = [f['text_length'] for f in non_rumor_data]
    print(f"平均文本长度: 谣言 {statistics.mean(rumor_text_lengths):.1f}, 非谣言 {statistics.mean(non_rumor_text_lengths):.1f}")

def find_propagation_outliers(features_data, feature='virality_score', threshold_percentile=95):
    """找出传播异常值"""
    print(f"\n=== {feature} 异常值分析 ===")
    
    values = [f[feature] for f in features_data if feature in f]
    threshold = sorted(values)[int(len(values) * threshold_percentile / 100)]
    
    outliers = [f for f in features_data if f[feature] >= threshold]
    
    print(f"阈值 (第{threshold_percentile}百分位): {threshold:.4f}")
    print(f"异常值数量: {len(outliers)}")
    
    # 分析异常值特征
    rumor_outliers = [f for f in outliers if f['is_rumor']]
    non_rumor_outliers = [f for f in outliers if not f['is_rumor']]
    
    print(f"谣言异常值: {len(rumor_outliers)} ({len(rumor_outliers)/len(outliers)*100:.1f}%)")
    print(f"非谣言异常值: {len(non_rumor_outliers)} ({len(non_rumor_outliers)/len(outliers)*100:.1f}%)")
    
    return outliers

def analyze_temporal_patterns(features_data):
    """分析时间传播模式"""
    print("\n=== 时间传播模式分析 ===")
    
    # 早期传播强度分析
    early_intensities = [f['early_propagation_intensity'] for f in features_data]
    print(f"早期传播强度 (前6小时):")
    print(f"  平均: {statistics.mean(early_intensities):.2f}")
    print(f"  中位数: {statistics.median(early_intensities):.2f}")
    print(f"  最大: {max(early_intensities)}")
    
    # 传播持续时间分析
    durations = [f['propagation_duration'] for f in features_data if f['propagation_duration'] > 0]
    if durations:
        print(f"\n传播持续时间 (小时):")
        print(f"  平均: {statistics.mean(durations):.2f}")
        print(f"  中位数: {statistics.median(durations):.2f}")
        print(f"  最长: {max(durations):.2f}")
    
    # 快速传播vs慢速传播
    speeds = [f['propagation_speed'] for f in features_data if f['propagation_speed'] > 0]
    if speeds:
        median_speed = statistics.median(speeds)
        fast_propagation = [f for f in features_data if f['propagation_speed'] > median_speed]
        slow_propagation = [f for f in features_data if f['propagation_speed'] <= median_speed and f['propagation_speed'] > 0]
        
        print(f"\n传播速度分析 (以中位数 {median_speed:.2f} 为界):")
        print(f"  快速传播: {len(fast_propagation)} 条")
        print(f"  慢速传播: {len(slow_propagation)} 条")
        
        # 快速传播中谣言比例
        fast_rumors = sum(1 for f in fast_propagation if f['is_rumor'])
        slow_rumors = sum(1 for f in slow_propagation if f['is_rumor'])
        
        print(f"  快速传播中谣言比例: {fast_rumors/len(fast_propagation)*100:.1f}%")
        print(f"  慢速传播中谣言比例: {slow_rumors/len(slow_propagation)*100:.1f}%")

def create_feature_correlation_analysis(features_data):
    """创建特征相关性分析"""
    print("\n=== 特征相关性分析 ===")
    
    numeric_features = [
        'forward_count', 'like_count', 'reply_count', 
        'propagation_depth', 'cascade_size', 'virality_score',
        'propagation_speed', 'text_length'
    ]
    
    # 计算简单的相关性
    print("主要特征与病毒性分数的关系:")
    virality_scores = [f['virality_score'] for f in features_data]
    
    for feature in numeric_features[:-1]:  # 除了virality_score本身
        feature_values = [f[feature] for f in features_data]
        
        # 计算皮尔逊相关系数的简化版本
        mean_v = statistics.mean(virality_scores)
        mean_f = statistics.mean(feature_values)
        
        numerator = sum((v - mean_v) * (f - mean_f) for v, f in zip(virality_scores, feature_values))
        denom_v = sum((v - mean_v) ** 2 for v in virality_scores)
        denom_f = sum((f - mean_f) ** 2 for f in feature_values)
        
        if denom_v > 0 and denom_f > 0:
            correlation = numerator / math.sqrt(denom_v * denom_f)
            print(f"  {feature}: {correlation:.3f}")

def generate_insights(features_data):
    """生成数据洞察"""
    print("\n=== 数据洞察 ===")
    
    rumor_data = [f for f in features_data if f['is_rumor']]
    non_rumor_data = [f for f in features_data if not f['is_rumor']]
    
    # 洞察1: 传播深度vs级联大小
    rumor_avg_depth = statistics.mean([f['propagation_depth'] for f in rumor_data])
    non_rumor_avg_depth = statistics.mean([f['propagation_depth'] for f in non_rumor_data])
    
    rumor_avg_cascade = statistics.mean([f['cascade_size'] for f in rumor_data])
    non_rumor_avg_cascade = statistics.mean([f['cascade_size'] for f in non_rumor_data])
    
    print(f"1. 传播模式差异:")
    print(f"   - 谣言倾向于更深层传播 (深度: {rumor_avg_depth:.2f} vs {non_rumor_avg_depth:.2f})")
    print(f"   - 非谣言倾向于更广泛传播 (级联大小: {rumor_avg_cascade:.2f} vs {non_rumor_avg_cascade:.2f})")
    
    # 洞察2: 早期传播强度
    rumor_early = statistics.mean([f['early_propagation_intensity'] for f in rumor_data])
    non_rumor_early = statistics.mean([f['early_propagation_intensity'] for f in non_rumor_data])
    
    print(f"\n2. 早期传播特征:")
    print(f"   - 谣言早期传播强度: {rumor_early:.2f}")
    print(f"   - 非谣言早期传播强度: {non_rumor_early:.2f}")
    print(f"   - {'谣言' if rumor_early > non_rumor_early else '非谣言'}在早期传播更活跃")
    
    # 洞察3: 互动模式
    rumor_like_ratio = statistics.mean([f['like_forward_ratio'] for f in rumor_data])
    non_rumor_like_ratio = statistics.mean([f['like_forward_ratio'] for f in non_rumor_data])
    
    rumor_reply_ratio = statistics.mean([f['reply_forward_ratio'] for f in rumor_data])
    non_rumor_reply_ratio = statistics.mean([f['reply_forward_ratio'] for f in non_rumor_data])
    
    print(f"\n3. 互动模式:")
    print(f"   - 谣言点赞转发比: {rumor_like_ratio:.3f}")
    print(f"   - 非谣言点赞转发比: {non_rumor_like_ratio:.3f}")
    print(f"   - 谣言回复转发比: {rumor_reply_ratio:.3f}")
    print(f"   - 非谣言回复转发比: {non_rumor_reply_ratio:.3f}")

def create_summary_report(features_data, output_dir="/workspace/results"):
    """创建总结报告"""
    report_file = f"{output_dir}/propagation_analysis_report.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== 中文谣言数据集传播特征分析报告 ===\n\n")
        
        # 基本统计
        total_count = len(features_data)
        rumor_count = sum(1 for f in features_data if f['is_rumor'])
        non_rumor_count = total_count - rumor_count
        
        f.write(f"数据集概况:\n")
        f.write(f"- 总样本数: {total_count}\n")
        f.write(f"- 谣言数: {rumor_count} ({rumor_count/total_count*100:.1f}%)\n")
        f.write(f"- 非谣言数: {non_rumor_count} ({non_rumor_count/total_count*100:.1f}%)\n\n")
        
        # 传播特征统计
        f.write("传播特征统计:\n")
        key_features = ['forward_count', 'like_count', 'reply_count', 'propagation_depth', 
                       'cascade_size', 'virality_score', 'propagation_speed']
        
        for feature in key_features:
            values = [f[feature] for f in features_data if feature in f]
            if values:
                f.write(f"- {feature}:\n")
                f.write(f"  平均值: {statistics.mean(values):.2f}\n")
                f.write(f"  中位数: {statistics.median(values):.2f}\n")
                f.write(f"  最大值: {max(values):.2f}\n")
                f.write(f"  最小值: {min(values):.2f}\n")
        
        f.write("\n谣言vs非谣言关键差异:\n")
        rumor_data = [f for f in features_data if f['is_rumor']]
        non_rumor_data = [f for f in features_data if not f['is_rumor']]
        
        for feature in key_features:
            rumor_values = [f[feature] for f in rumor_data if feature in f]
            non_rumor_values = [f[feature] for f in non_rumor_data if feature in f]
            
            if rumor_values and non_rumor_values:
                rumor_mean = statistics.mean(rumor_values)
                non_rumor_mean = statistics.mean(non_rumor_values)
                diff_pct = (rumor_mean - non_rumor_mean) / non_rumor_mean * 100
                
                f.write(f"- {feature}: 谣言比非谣言{'高' if diff_pct > 0 else '低'} {abs(diff_pct):.1f}%\n")
    
    print(f"分析报告已保存到: {report_file}")

def main():
    """主函数"""
    print("=== 传播特征分析开始 ===")
    
    # 加载数据
    try:
        features_data = load_features()
        print(f"成功加载 {len(features_data)} 条记录")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 执行各种分析
    analyze_feature_distributions(features_data)
    analyze_high_virality_content(features_data)
    analyze_propagation_patterns(features_data)
    analyze_text_features(features_data)
    find_propagation_outliers(features_data)
    create_feature_correlation_analysis(features_data)
    generate_insights(features_data)
    create_summary_report(features_data)
    
    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    main()