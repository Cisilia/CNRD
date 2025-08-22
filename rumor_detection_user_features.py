"""
中文谣言数据集用户特征提取器
Chinese Rumor Dataset User Feature Extractor

基于Chinese_Rumor_Dataset-master数据集的用户特征提取，用于谣言检测模型
User feature extraction for rumor detection models based on Chinese_Rumor_Dataset-master
"""

import json
import numpy as np
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
import pandas as pd
from pathlib import Path

class RumorDetectionUserFeatures:
    """
    谣言检测用户特征提取器
    User Feature Extractor for Rumor Detection
    """
    
    def __init__(self, reference_time: Optional[float] = None):
        """
        初始化特征提取器
        
        Args:
            reference_time: 参考时间戳，用于计算账户年龄。如果为None，使用当前时间
        """
        self.reference_time = reference_time or datetime.now().timestamp()
    
    def extract_basic_user_features(self, user_data: Dict, post_time: Optional[int] = None) -> List[float]:
        """
        提取基础用户特征（按照用户要求的格式）
        Extract basic user features (following user's requested format)
        
        Args:
            user_data: 用户数据字典
            post_time: 发布时间戳
            
        Returns:
            features: [粉丝数, 关注数, 是否认证(0/1), 账户年龄, 历史推文数]
        """
        # 基础特征提取
        follower_count = user_data.get('followers', 0)      # 粉丝数
        friend_count = user_data.get('friends', 0)          # 关注数
        verified = 1 if user_data.get('verified', False) else 0  # 是否认证 (0/1)
        tweet_count = user_data.get('messages', 0)          # 历史推文数
        
        # 账户年龄计算
        account_creation_time = user_data.get('time', 0)
        if account_creation_time > 0:
            reference_time = post_time if post_time else self.reference_time
            account_age = (reference_time - account_creation_time) / (24 * 3600)  # 转换为天数
            account_age = max(0, account_age)
        else:
            account_age = 0
        
        features = [
            follower_count,      # 粉丝数
            friend_count,        # 关注数
            verified,            # 是否认证 (0/1)
            account_age,         # 账户年龄
            tweet_count,         # 历史推文数
        ]
        
        return features
    
    def extract_advanced_user_features(self, user_data: Dict, post_time: Optional[int] = None) -> List[float]:
        """
        提取高级用户特征（包含更多衍生特征）
        Extract advanced user features (including more derived features)
        """
        # 首先获取基础特征
        basic_features = self.extract_basic_user_features(user_data, post_time)
        follower_count, friend_count, verified, account_age, tweet_count = basic_features
        
        # 高级衍生特征
        # 1. 社交网络比例特征
        if friend_count > 0:
            follower_friend_ratio = follower_count / friend_count
        else:
            follower_friend_ratio = follower_count
        
        # 2. 活跃度特征
        if account_age > 0:
            avg_tweets_per_day = tweet_count / account_age
        else:
            avg_tweets_per_day = 0
        
        # 3. 影响力特征
        social_influence = np.log1p(follower_count) + verified * 2
        
        # 4. 网络密度特征
        network_density = np.log1p(follower_count + friend_count)
        
        # 5. 信誉特征
        credibility_score = (
            verified * 0.4 +
            np.log1p(follower_count) * 0.3 +
            (1 if account_age > 365 else 0) * 0.3
        )
        
        # 6. 行为特征
        has_description = 1 if user_data.get('description', False) else 0
        verified_type = user_data.get('verified_type', -1)
        
        # 7. 地理和人口统计特征
        location = user_data.get('location', '')
        has_location = 1 if location and location.strip() != '' and location != '其他' else 0
        
        gender = user_data.get('gender', '')
        gender_male = 1 if gender == 'm' else 0
        gender_female = 1 if gender == 'f' else 0
        
        # 8. 互动潜力特征
        interaction_potential = np.sqrt(follower_count * friend_count)
        
        # 组合所有特征
        advanced_features = basic_features + [
            follower_friend_ratio,     # 粉丝关注比
            avg_tweets_per_day,        # 平均每日推文数
            social_influence,          # 社交影响力
            network_density,           # 网络密度
            credibility_score,         # 信誉分数
            has_description,           # 是否有描述
            verified_type,             # 认证类型
            has_location,              # 是否有位置
            gender_male,               # 是否男性
            gender_female,             # 是否女性
            interaction_potential,     # 互动潜力
        ]
        
        return advanced_features
    
    def get_basic_feature_names(self) -> List[str]:
        """获取基础特征名称"""
        return [
            'follower_count',    # 粉丝数
            'friend_count',      # 关注数
            'verified',          # 是否认证
            'account_age',       # 账户年龄
            'tweet_count',       # 历史推文数
        ]
    
    def get_advanced_feature_names(self) -> List[str]:
        """获取高级特征名称"""
        basic_names = self.get_basic_feature_names()
        advanced_names = basic_names + [
            'follower_friend_ratio',   # 粉丝关注比
            'avg_tweets_per_day',      # 平均每日推文数
            'social_influence',        # 社交影响力
            'network_density',         # 网络密度
            'credibility_score',       # 信誉分数
            'has_description',         # 是否有描述
            'verified_type',           # 认证类型
            'has_location',            # 是否有位置
            'gender_male',             # 是否男性
            'gender_female',           # 是否女性
            'interaction_potential',   # 互动潜力
        ]
        return advanced_names

def load_chinese_rumor_dataset(dataset_path: str, use_advanced_features: bool = False) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    加载中文谣言数据集并提取用户特征
    Load Chinese rumor dataset and extract user features
    
    Args:
        dataset_path: 数据集路径
        use_advanced_features: 是否使用高级特征
        
    Returns:
        user_features: 用户特征矩阵 [N, user_feature_dim]
        user_ids: 用户ID列表
        feature_names: 特征名称列表
    """
    extractor = RumorDetectionUserFeatures()
    user_features = []
    user_ids = []
    
    print(f"正在从 {dataset_path} 加载数据...")
    
    # 遍历数据集目录
    json_files = [f for f in os.listdir(dataset_path) if f.endswith('.json')]
    print(f"找到 {len(json_files)} 个JSON文件")
    
    processed_count = 0
    for filename in json_files:
        file_path = os.path.join(dataset_path, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            user_info = data.get('user')
            if not isinstance(user_info, dict):
                continue
            
            # 提取用户ID
            user_id = filename.split('_')[2].split('.')[0]
            user_ids.append(user_id)
            
            # 提取特征
            if use_advanced_features:
                features = extractor.extract_advanced_user_features(user_info, data.get('time'))
            else:
                features = extractor.extract_basic_user_features(user_info, data.get('time'))
            
            user_features.append(features)
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"已处理 {processed_count} 个文件...")
                
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
            continue
    
    # 转换为numpy数组
    user_features = np.array(user_features)  # [N, user_feature_dim]
    
    # 获取特征名称
    if use_advanced_features:
        feature_names = extractor.get_advanced_feature_names()
    else:
        feature_names = extractor.get_basic_feature_names()
    
    print(f"\n特征提取完成!")
    print(f"用户数量: {len(user_ids)}")
    print(f"特征维度: {user_features.shape[1] if len(user_features) > 0 else 0}")
    print(f"特征名称: {feature_names}")
    
    return user_features, user_ids, feature_names

# 按照用户要求的格式实现
def extract_user_features_user_format(dataset_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    完全按照用户要求的代码格式提取特征
    Extract features exactly in the format requested by the user
    """
    extractor = RumorDetectionUserFeatures()
    
    # 加载所有用户数据
    user_data_dict = {}
    user_ids = []
    
    for filename in os.listdir(dataset_path):
        if filename.endswith('.json'):
            file_path = os.path.join(dataset_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                user_info = data.get('user')
                if isinstance(user_info, dict):
                    user_id = filename.split('_')[2].split('.')[0]
                    user_data_dict[user_id] = {
                        'user_info': user_info,
                        'post_time': data.get('time')
                    }
                    user_ids.append(user_id)
            except Exception as e:
                continue
    
    # 按照用户要求的格式提取特征
    user_features = []
    for user_id in user_ids:
        user = user_data_dict[user_id]['user_info']
        post_time = user_data_dict[user_id]['post_time']
        
        # 计算账户年龄
        account_creation_time = user.get('time', 0)
        if account_creation_time > 0 and post_time:
            account_age = (post_time - account_creation_time) / (24 * 3600)
            account_age = max(0, account_age)
        else:
            account_age = 0
        
        features = [
            user.get('followers', 0),      # follower_count - 粉丝数
            user.get('friends', 0),        # friend_count - 关注数
            1 if user.get('verified', False) else 0,  # verified - 是否认证 (0/1)
            account_age,                   # account_age - 账户年龄
            user.get('messages', 0),       # tweet_count - 历史推文数
        ]
        user_features.append(features)
    
    user_features = np.array(user_features)  # [N, user_feature_dim]
    
    return user_features, user_ids

# 测试和使用示例
if __name__ == "__main__":
    dataset_path = "/workspace/CED_Dataset/original-microblog"
    
    print("=== 基础特征提取（用户要求格式） ===")
    user_features, user_ids = extract_user_features_user_format(dataset_path)
    
    print(f"用户特征矩阵形状: {user_features.shape}")
    print(f"特征维度说明: [粉丝数, 关注数, 是否认证(0/1), 账户年龄(天), 历史推文数]")
    
    if len(user_features) > 0:
        print(f"\n特征统计:")
        print(f"粉丝数 - 平均: {np.mean(user_features[:, 0]):.2f}, 最大: {np.max(user_features[:, 0])}")
        print(f"关注数 - 平均: {np.mean(user_features[:, 1]):.2f}, 最大: {np.max(user_features[:, 1])}")
        print(f"认证用户比例: {np.mean(user_features[:, 2]):.2%}")
        print(f"平均账户年龄: {np.mean(user_features[:, 3]):.2f} 天")
        print(f"平均推文数: {np.mean(user_features[:, 4]):.2f}")
    
    print("\n=== 高级特征提取 ===")
    user_features_advanced, user_ids_advanced, feature_names_advanced = load_chinese_rumor_dataset(
        dataset_path, use_advanced_features=True
    )
    
    # 保存结果
    np.save('/workspace/basic_user_features.npy', user_features)
    np.save('/workspace/advanced_user_features.npy', user_features_advanced)
    
    with open('/workspace/user_ids.json', 'w', encoding='utf-8') as f:
        json.dump(user_ids, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存:")
    print(f"- 基础特征: /workspace/basic_user_features.npy")
    print(f"- 高级特征: /workspace/advanced_user_features.npy") 
    print(f"- 用户ID: /workspace/user_ids.json")