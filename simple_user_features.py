import json
import numpy as np
import os
from datetime import datetime

def extract_user_features_simple(dataset_path: str):
    """
    简化版用户特征提取，匹配用户要求的格式
    Simplified user feature extraction matching the user's requested format
    """
    user_features = []
    user_ids = []
    
    # 遍历数据集目录中的所有JSON文件
    for filename in os.listdir(dataset_path):
        if filename.endswith('.json'):
            file_path = os.path.join(dataset_path, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                user_info = data.get('user')
                if not isinstance(user_info, dict):
                    continue
                
                # 提取用户ID（从文件名中获取）
                user_id = filename.split('_')[2].split('.')[0]  # 从文件名格式中提取用户ID
                user_ids.append(user_id)
                
                # 计算账户年龄
                account_creation_time = user_info.get('time', 0)
                post_time = data.get('time', datetime.now().timestamp())
                if account_creation_time > 0:
                    account_age_days = (post_time - account_creation_time) / (24 * 3600)
                    account_age_days = max(0, account_age_days)
                else:
                    account_age_days = 0
                
                # 按照用户要求的格式提取特征
                features = [
                    user_info.get('followers', 0),      # 粉丝数 (follower_count)
                    user_info.get('friends', 0),        # 关注数 (friend_count)
                    1 if user_info.get('verified', False) else 0,  # 是否认证 (verified)
                    account_age_days,                   # 账户年龄 (account_age)
                    user_info.get('messages', 0),       # 历史推文数 (tweet_count)
                ]
                
                user_features.append(features)
                
            except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
                print(f"Error processing file {file_path}: {e}")
                continue
    
    # 转换为numpy数组，匹配用户要求的格式
    user_features = np.array(user_features)  # [N, user_feature_dim]
    
    print(f"提取了 {len(user_ids)} 个用户的特征")
    print(f"特征维度: {user_features.shape}")
    print(f"特征说明: [粉丝数, 关注数, 是否认证(0/1), 账户年龄(天), 历史推文数]")
    
    return user_features, user_ids

# 使用示例 - 完全按照用户要求的格式
if __name__ == "__main__":
    # 数据集路径
    dataset_path = "/workspace/CED_Dataset/original-microblog"
    
    # 提取用户特征 - 按照用户要求的格式
    user_features, user_ids = extract_user_features_simple(dataset_path)
    
    # 用户要求的代码格式示例：
    """
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
    
    print(f"\n最终用户特征矩阵: {user_features.shape}")
    print(f"用户ID列表长度: {len(user_ids)}")
    
    # 保存结果
    np.save('/workspace/user_features_simple.npy', user_features)
    
    with open('/workspace/user_ids.json', 'w', encoding='utf-8') as f:
        json.dump(user_ids, f, ensure_ascii=False, indent=2)
    
    print("简化版用户特征已保存到: /workspace/user_features_simple.npy")
    print("用户ID列表已保存到: /workspace/user_ids.json")