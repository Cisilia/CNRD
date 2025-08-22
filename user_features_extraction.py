import json
import numpy as np
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import pandas as pd

class UserFeatureExtractor:
    """
    用户特征提取器，基于中文谣言数据集
    User Feature Extractor for Chinese Rumor Dataset
    """
    
    def __init__(self):
        self.current_timestamp = datetime.now().timestamp()
    
    def extract_user_features_from_file(self, file_path: str) -> Optional[List[float]]:
        """
        从单个JSON文件中提取用户特征
        Extract user features from a single JSON file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            user_info = data.get('user')
            if not isinstance(user_info, dict):
                return None
            
            return self._extract_features_from_user_dict(user_info, data.get('time'))
        
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"Error processing file {file_path}: {e}")
            return None
    
    def _extract_features_from_user_dict(self, user_info: Dict, post_time: Optional[int] = None) -> List[float]:
        """
        从用户字典中提取特征
        Extract features from user dictionary
        """
        # 基础用户特征 (Basic user features)
        followers_count = user_info.get('followers', 0)  # 粉丝数
        friends_count = user_info.get('friends', 0)      # 关注数 
        verified = 1 if user_info.get('verified', False) else 0  # 是否认证
        messages_count = user_info.get('messages', 0)    # 历史推文数
        
        # 账户年龄计算 (Account age calculation)
        account_creation_time = user_info.get('time', 0)
        if account_creation_time > 0:
            # 使用发布时间作为参考点，如果没有则使用当前时间
            reference_time = post_time if post_time else self.current_timestamp
            account_age_days = (reference_time - account_creation_time) / (24 * 3600)
            account_age_days = max(0, account_age_days)  # 确保非负
        else:
            account_age_days = 0
        
        # 衍生特征 (Derived features)
        # 粉丝关注比 (Follower-to-friend ratio)
        if friends_count > 0:
            follower_friend_ratio = followers_count / friends_count
        else:
            follower_friend_ratio = followers_count  # 如果关注数为0，比值等于粉丝数
        
        # 平均每日发帖数 (Average posts per day)
        if account_age_days > 0:
            avg_posts_per_day = messages_count / account_age_days
        else:
            avg_posts_per_day = 0
        
        # 影响力指标 (Influence metrics)
        # 社交网络密度指标
        social_network_density = np.log1p(followers_count + friends_count)
        
        # 活跃度指标
        activity_score = np.log1p(messages_count) * np.log1p(account_age_days + 1)
        
        # 权威性指标 (Authority score)
        authority_score = verified * 2 + np.log1p(followers_count) * 0.1
        
        # 额外的行为特征 (Additional behavioral features)
        has_description = 1 if user_info.get('description', False) else 0
        verified_type = user_info.get('verified_type', -1)
        
        # 地理位置特征 (Location features)
        location = user_info.get('location', '')
        has_location = 1 if location and location.strip() != '' and location != '其他' else 0
        
        # 性别特征 (Gender features) 
        gender = user_info.get('gender', '')
        gender_male = 1 if gender == 'm' else 0
        gender_female = 1 if gender == 'f' else 0
        
        features = [
            # 基础特征 (Basic features)
            followers_count,           # 粉丝数
            friends_count,             # 关注数
            verified,                  # 是否认证 (0/1)
            account_age_days,          # 账户年龄（天）
            messages_count,            # 历史推文数
            
            # 衍生特征 (Derived features)
            follower_friend_ratio,     # 粉丝关注比
            avg_posts_per_day,         # 平均每日发帖数
            social_network_density,    # 社交网络密度
            activity_score,            # 活跃度分数
            authority_score,           # 权威性分数
            
            # 额外特征 (Additional features)
            has_description,           # 是否有个人描述
            verified_type,             # 认证类型
            has_location,              # 是否有位置信息
            gender_male,               # 是否男性
            gender_female,             # 是否女性
        ]
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称列表
        Get feature names list
        """
        return [
            'followers_count',         # 粉丝数
            'friends_count',           # 关注数
            'verified',                # 是否认证
            'account_age_days',        # 账户年龄（天）
            'messages_count',          # 历史推文数
            'follower_friend_ratio',   # 粉丝关注比
            'avg_posts_per_day',       # 平均每日发帖数
            'social_network_density',  # 社交网络密度
            'activity_score',          # 活跃度分数
            'authority_score',         # 权威性分数
            'has_description',         # 是否有个人描述
            'verified_type',           # 认证类型
            'has_location',            # 是否有位置信息
            'gender_male',             # 是否男性
            'gender_female',           # 是否女性
        ]
    
    def extract_features_from_directory(self, directory_path: str) -> Tuple[np.ndarray, List[str]]:
        """
        从目录中的所有JSON文件提取用户特征
        Extract user features from all JSON files in a directory
        """
        features_list = []
        file_names = []
        
        for filename in os.listdir(directory_path):
            if filename.endswith('.json'):
                file_path = os.path.join(directory_path, filename)
                features = self.extract_user_features_from_file(file_path)
                if features is not None:
                    features_list.append(features)
                    file_names.append(filename)
        
        if not features_list:
            return np.array([]), []
        
        return np.array(features_list), file_names
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        特征标准化
        Normalize features using z-score normalization
        """
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        return scaler.fit_transform(features)
    
    def get_feature_statistics(self, features: np.ndarray) -> pd.DataFrame:
        """
        获取特征统计信息
        Get feature statistics
        """
        feature_names = self.get_feature_names()
        stats_df = pd.DataFrame({
            'feature': feature_names,
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'min': np.min(features, axis=0),
            'max': np.max(features, axis=0),
            'median': np.median(features, axis=0)
        })
        return stats_df


def extract_user_features_for_rumor_detection(dataset_path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    主函数：为谣言检测提取用户特征
    Main function: Extract user features for rumor detection
    
    Args:
        dataset_path: 数据集路径 (path to dataset directory)
    
    Returns:
        user_features: 用户特征矩阵 [N, feature_dim]
        feature_names: 特征名称列表
        file_names: 对应的文件名列表
    """
    extractor = UserFeatureExtractor()
    
    # 提取特征
    user_features, file_names = extractor.extract_features_from_directory(dataset_path)
    feature_names = extractor.get_feature_names()
    
    print(f"提取了 {len(file_names)} 个用户的特征")
    print(f"特征维度: {user_features.shape[1] if len(user_features) > 0 else 0}")
    print(f"特征名称: {feature_names}")
    
    return user_features, feature_names, file_names


# 使用示例 (Usage example)
if __name__ == "__main__":
    # 设置数据集路径
    original_microblog_path = "/workspace/CED_Dataset/original-microblog"
    
    # 提取用户特征
    user_features, feature_names, file_names = extract_user_features_for_rumor_detection(original_microblog_path)
    
    if len(user_features) > 0:
        print(f"\n用户特征矩阵形状: {user_features.shape}")
        print(f"特征名称: {feature_names}")
        
        # 显示特征统计
        extractor = UserFeatureExtractor()
        stats = extractor.get_feature_statistics(user_features)
        print("\n特征统计信息:")
        print(stats)
        
        # 保存特征
        np.save('/workspace/user_features.npy', user_features)
        print(f"\n用户特征已保存到: /workspace/user_features.npy")
        
        # 保存特征名称和文件对应关系
        metadata = {
            'feature_names': feature_names,
            'file_names': file_names,
            'feature_descriptions': {
                'followers_count': '粉丝数',
                'friends_count': '关注数',
                'verified': '是否认证',
                'account_age_days': '账户年龄（天）',
                'messages_count': '历史推文数',
                'follower_friend_ratio': '粉丝关注比',
                'avg_posts_per_day': '平均每日发帖数',
                'social_network_density': '社交网络密度',
                'activity_score': '活跃度分数',
                'authority_score': '权威性分数',
                'has_description': '是否有个人描述',
                'verified_type': '认证类型',
                'has_location': '是否有位置信息',
                'gender_male': '是否男性',
                'gender_female': '是否女性'
            }
        }
        
        with open('/workspace/user_features_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print("元数据已保存到: /workspace/user_features_metadata.json")
    else:
        print("未找到有效的用户数据")