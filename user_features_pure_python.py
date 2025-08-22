"""
中文谣言数据集用户特征提取器 - 纯Python版本
Chinese Rumor Dataset User Feature Extractor - Pure Python Version

不依赖外部库，仅使用Python标准库
No external dependencies, only Python standard library
"""

import json
import os
import math
from datetime import datetime
from typing import List, Dict, Tuple, Optional

class UserFeatureExtractorPure:
    """
    纯Python用户特征提取器
    Pure Python User Feature Extractor
    """
    
    def __init__(self, reference_time: Optional[float] = None):
        self.reference_time = reference_time or datetime.now().timestamp()
    
    def extract_user_features(self, user_data: Dict, post_time: Optional[int] = None) -> List[float]:
        """
        提取用户特征 - 完全按照用户要求的格式
        Extract user features - exactly following user's requested format
        
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

def extract_user_features_from_dataset(dataset_path: str) -> Tuple[List[List[float]], List[str]]:
    """
    从数据集中提取用户特征 - 按照用户要求的格式
    Extract user features from dataset - following user's requested format
    
    完全实现用户要求的代码结构：
    user_features = []
    for user_id in user_ids:
        features = [
            user.follower_count,      # 粉丝数
            user.friend_count,        # 关注数
            user.verified,            # 是否认证 (0/1)
            user.account_age,         # 账户年龄
            user.tweet_count,         # 历史推文数
        ]
        user_features.append(features)
    user_features = np.array(user_features)  # [N, user_feature_dim]
    """
    
    extractor = UserFeatureExtractorPure()
    
    # 首先收集所有用户数据
    user_data_dict = {}
    user_ids = []
    
    print(f"正在从 {dataset_path} 加载数据...")
    
    # 遍历数据集目录
    json_files = [f for f in os.listdir(dataset_path) if f.endswith('.json')]
    print(f"找到 {len(json_files)} 个JSON文件")
    
    for filename in json_files:
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
            print(f"处理文件 {filename} 时出错: {e}")
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
            user.get('followers', 0),      # user.follower_count - 粉丝数
            user.get('friends', 0),        # user.friend_count - 关注数
            1 if user.get('verified', False) else 0,  # user.verified - 是否认证 (0/1)
            account_age,                   # user.account_age - 账户年龄
            user.get('messages', 0),       # user.tweet_count - 历史推文数
        ]
        user_features.append(features)
    
    # 这里应该是 np.array(user_features)，但由于没有numpy，我们返回列表
    # user_features = np.array(user_features)  # [N, user_feature_dim]
    
    print(f"\n特征提取完成!")
    print(f"用户数量: {len(user_ids)}")
    print(f"特征维度: {len(user_features[0]) if user_features else 0}")
    print(f"特征说明: [粉丝数, 关注数, 是否认证(0/1), 账户年龄(天), 历史推文数]")
    
    return user_features, user_ids

def convert_to_numpy_format(user_features: List[List[float]]) -> str:
    """
    生成转换为numpy格式的代码
    Generate code to convert to numpy format
    """
    return """
# 转换为numpy数组格式（需要安装numpy）
import numpy as np
user_features = np.array(user_features)  # [N, user_feature_dim]
"""

def print_feature_statistics(user_features: List[List[float]]):
    """
    打印特征统计信息（纯Python实现）
    Print feature statistics (pure Python implementation)
    """
    if not user_features:
        print("没有用户特征数据")
        return
    
    n_users = len(user_features)
    n_features = len(user_features[0])
    
    print(f"\n=== 用户特征统计 ===")
    print(f"用户数量: {n_users}")
    print(f"特征维度: {n_features}")
    
    feature_names = ['粉丝数', '关注数', '是否认证', '账户年龄(天)', '历史推文数']
    
    for i in range(n_features):
        values = [features[i] for features in user_features]
        mean_val = sum(values) / len(values)
        max_val = max(values)
        min_val = min(values)
        
        print(f"{feature_names[i]}: 平均={mean_val:.2f}, 最大={max_val}, 最小={min_val}")

# 使用示例
if __name__ == "__main__":
    dataset_path = "/workspace/CED_Dataset/original-microblog"
    
    print("=== 中文谣言数据集用户特征提取 ===")
    
    # 提取用户特征
    user_features, user_ids = extract_user_features_from_dataset(dataset_path)
    
    # 显示统计信息
    print_feature_statistics(user_features)
    
    # 保存结果（JSON格式，便于后续加载）
    result_data = {
        'user_features': user_features,
        'user_ids': user_ids,
        'feature_names': ['follower_count', 'friend_count', 'verified', 'account_age', 'tweet_count'],
        'feature_descriptions': ['粉丝数', '关注数', '是否认证(0/1)', '账户年龄(天)', '历史推文数']
    }
    
    with open('/workspace/user_features_data.json', 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: /workspace/user_features_data.json")
    
    # 生成numpy转换代码
    print(convert_to_numpy_format(user_features))
    
    # 显示用户要求的代码格式示例
    print("\n=== 用户要求的代码格式实现 ===")
    print("""
# 按照您要求的格式：
user_features = []
for user_id in user_ids:
    features = [
        user.follower_count,      # 粉丝数
        user.friend_count,        # 关注数
        user.verified,            # 是否认证 (0/1)
        user.account_age,         # 账户年龄
        user.tweet_count,         # 历史推文数
    ]
    user_features.append(features)
user_features = np.array(user_features)  # [N, user_feature_dim]

# 已经实现在 extract_user_features_from_dataset() 函数中
    """)