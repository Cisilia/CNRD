# 中文谣言数据集传播特征构建项目

## 项目概述

本项目基于Chinese_Rumor_Dataset-master数据集，构建了全面的传播特征，包括转发数、点赞数、回复数、传播深度、级联大小、距离根节点时间、病毒性分数等关键指标。

## 数据集信息

- **总微博数**: 3,387条
- **谣言数**: 1,538条 (45.4%)
- **非谣言数**: 1,849条 (54.6%)
- **数据来源**: 新浪微博不实信息举报平台
- **时间范围**: 2009年9月4日至2017年6月12日

## 构建的传播特征

### 基础传播特征
1. **转发数 (forward_count)**: 实际转发次数
2. **点赞数 (like_count)**: 获得的点赞数量
3. **回复数 (reply_count)**: 收到的回复数量
4. **传播深度 (propagation_depth)**: 传播的最大层级深度
5. **级联大小 (cascade_size)**: 总传播节点数量

### 时间特征
6. **距离根节点时间 (avg_time_from_root)**: 平均传播时间距离（小时）
7. **传播持续时间 (propagation_duration)**: 从开始到结束的传播时长
8. **早期传播强度 (early_propagation_intensity)**: 前6小时的传播量

### 计算特征
9. **病毒性分数 (virality_score)**: 综合多个指标的传播病毒性评分
10. **传播效率 (propagation_efficiency)**: 级联大小与传播深度的比值
11. **传播速度 (propagation_speed)**: 级联大小与平均传播时间的比值
12. **互动率 (interaction_rate)**: 总互动数与级联大小的比值
13. **点赞转发比 (like_forward_ratio)**: 点赞数与转发数的比值
14. **回复转发比 (reply_forward_ratio)**: 回复数与转发数的比值

### 文本特征
15. **文本长度 (text_length)**: 原微博文本字符数
16. **是否包含URL (has_url)**: 是否包含网址链接
17. **是否包含@提及 (has_mention)**: 是否包含用户提及
18. **是否包含话题标签 (has_hashtag)**: 是否包含#话题标签

## 主要发现

### 1. 传播模式差异
- **谣言特点**: 倾向于更深层传播（平均深度5.76 vs 4.23）
- **非谣言特点**: 倾向于更广泛传播（平均级联大小428.10 vs 314.45）

### 2. 早期传播特征
- 谣言在早期（前6小时）传播更活跃（强度8.45 vs 0.00）
- 这可能反映了谣言的紧急性和争议性特征

### 3. 互动模式
- 非谣言获得更多点赞（平均63.59 vs 10.81）
- 非谣言点赞转发比更高（0.416 vs 0.133）
- 非谣言更多使用URL（25.8% vs 10.4%）和话题标签（18.4% vs 7.7%）

### 4. 病毒性分数分布
- 平均病毒性分数: 3.79
- 非谣言平均病毒性更高（3.91 vs 3.65）
- 高病毒性内容（Top 95%）中85.3%为非谣言

## 文件说明

### 脚本文件
- `propagation_features_simple.py`: 基础版特征构建脚本
- `propagation_features_enhanced.py`: 增强版特征构建脚本（推荐使用）
- `analyze_propagation_features.py`: 特征分析脚本

### 结果文件
- `propagation_features_enhanced.csv`: 特征数据（CSV格式）
- `propagation_features_enhanced.json`: 特征数据（JSON格式）
- `statistics_enhanced.json`: 详细统计信息
- `feature_descriptions.txt`: 特征说明文档
- `propagation_analysis_report.txt`: 分析报告

## 使用方法

### 1. 构建传播特征
```bash
python3 propagation_features_enhanced.py
```

### 2. 分析传播特征
```bash
python3 analyze_propagation_features.py
```

### 3. 查看结果
- 特征数据: `results/propagation_features_enhanced.csv`
- 统计信息: `results/statistics_enhanced.json`
- 分析报告: `results/propagation_analysis_report.txt`

## 病毒性分数计算公式

病毒性分数综合考虑多个传播指标：

```
virality_score = 0.25 * log(1 + forward_count) +     # 转发权重
                 0.15 * log(1 + like_count) +         # 点赞权重  
                 0.20 * log(1 + reply_count) +        # 回复权重
                 0.20 * log(1 + cascade_size) +       # 级联大小权重
                 0.10 * log(1 + propagation_depth) +  # 传播深度权重
                 0.10 * time_factor                   # 时间因子权重
```

其中 `time_factor = 1.0 / (1.0 + avg_time_to_spread / 24.0)`

## 应用场景

1. **谣言检测**: 利用传播特征区分谣言和非谣言
2. **传播预测**: 基于早期特征预测传播规模
3. **内容推荐**: 根据病毒性分数优化内容推荐
4. **社交网络分析**: 研究信息在社交网络中的传播模式
5. **危机管理**: 识别和控制有害信息的传播

## 技术特点

- **纯Python实现**: 仅使用标准库，无外部依赖
- **高效处理**: 支持大规模数据集处理
- **完整特征**: 涵盖时间、结构、内容等多维度特征
- **详细分析**: 提供全面的统计分析和洞察

## 数据质量

- **处理成功率**: 100% (3,387/3,387)
- **时间解析**: 支持多种时间格式，包含中文日期
- **级联构建**: 准确构建传播树结构
- **特征完整性**: 所有样本都包含完整的特征集

## 引用

如果使用本项目的代码或结果，请引用原始数据集论文：

```bibtex
@article{song2018ced,
  title={CED: Credible Early Detection of Social Media Rumors},
  author={Song, Changhe and Tu, Cunchao and Yang, Cheng and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv preprint arXiv:1811.04175},
  year={2018}
}
```