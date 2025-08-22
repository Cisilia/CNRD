"""
完整的谣言检测用户特征集成示例
Complete example of integrating user features for rumor detection

展示如何将用户特征与现有的模型和文本特征结合使用
Shows how to combine user features with existing models and text features
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# 导入用户特征提取器
from user_features_pure_python import extract_user_features_from_dataset

def load_user_features_for_rumor_detection(dataset_path: str):
    """
    为谣言检测模型加载用户特征
    Load user features for rumor detection model
    
    完全按照用户要求的格式实现：
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
    
    print("=== 加载用户特征用于谣言检测 ===")
    
    # 提取用户特征
    user_features, user_ids = extract_user_features_from_dataset(dataset_path)
    
    # 转换为numpy数组（如果有numpy的话）
    try:
        import numpy as np
        user_features = np.array(user_features)  # [N, user_feature_dim]
        print(f"用户特征矩阵形状: {user_features.shape}")
    except ImportError:
        print("注意: 未安装numpy，返回Python列表格式")
        print(f"用户特征列表长度: {len(user_features)}")
        print(f"每个用户特征维度: {len(user_features[0]) if user_features else 0}")
    
    return user_features, user_ids

def create_user_profile_features_example():
    """
    创建用户画像特征的完整示例
    Create complete example of user profile features
    """
    
    # 按照用户要求的代码格式
    dataset_path = "/workspace/CED_Dataset/original-microblog"
    
    print("=== 用户画像特征提取示例 ===")
    print("按照您提供的代码格式实现：")
    print("""
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
    """)
    
    # 实际执行
    user_features, user_ids = load_user_features_for_rumor_detection(dataset_path)
    
    return user_features, user_ids

def integrate_with_existing_model_example(user_features, text_features=None):
    """
    与现有模型和文本特征集成的示例
    Example of integrating with existing model and text features
    """
    print("\n=== 与现有模型集成示例 ===")
    
    # 模拟文本特征（您说已经有了）
    if text_features is None:
        print("模拟文本特征（您已经有了文本特征的代码）")
        try:
            import numpy as np
            # 模拟文本特征矩阵 [N, text_feature_dim]
            n_users = len(user_features)
            text_feature_dim = 768  # 例如BERT特征维度
            text_features = np.random.randn(n_users, text_feature_dim)
            print(f"模拟文本特征形状: {text_features.shape}")
        except ImportError:
            print("无numpy，跳过文本特征模拟")
            text_features = None
    
    # 特征融合示例
    if text_features is not None:
        try:
            import numpy as np
            # 将用户特征和文本特征拼接
            if isinstance(user_features, list):
                user_features = np.array(user_features)
            
            combined_features = np.concatenate([text_features, user_features], axis=1)
            print(f"融合特征形状: {combined_features.shape}")
            print(f"文本特征维度: {text_features.shape[1]}")
            print(f"用户特征维度: {user_features.shape[1]}")
            print(f"总特征维度: {combined_features.shape[1]}")
            
            return combined_features
        except ImportError:
            print("无numpy，无法进行特征融合示例")
    
    return user_features

def save_features_for_model(user_features, user_ids, output_dir="/workspace"):
    """
    保存特征供模型使用
    Save features for model usage
    """
    print(f"\n=== 保存特征到 {output_dir} ===")
    
    # 保存用户特征
    feature_data = {
        'user_features': user_features,
        'user_ids': user_ids,
        'feature_names': ['follower_count', 'friend_count', 'verified', 'account_age', 'tweet_count'],
        'feature_descriptions': {
            'follower_count': '粉丝数',
            'friend_count': '关注数', 
            'verified': '是否认证 (0/1)',
            'account_age': '账户年龄（天）',
            'tweet_count': '历史推文数'
        },
        'dataset_info': {
            'total_users': len(user_ids),
            'feature_dim': len(user_features[0]) if user_features else 0,
            'extraction_time': datetime.now().isoformat()
        }
    }
    
    # 保存为JSON格式
    with open(os.path.join(output_dir, 'user_features_for_rumor_detection.json'), 'w', encoding='utf-8') as f:
        json.dump(feature_data, f, ensure_ascii=False, indent=2)
    
    # 如果有numpy，也保存为.npy格式
    try:
        import numpy as np
        if isinstance(user_features, list):
            user_features_array = np.array(user_features)
        else:
            user_features_array = user_features
        
        np.save(os.path.join(output_dir, 'user_features.npy'), user_features_array)
        print("用户特征已保存为:")
        print(f"- JSON格式: {output_dir}/user_features_for_rumor_detection.json")
        print(f"- NumPy格式: {output_dir}/user_features.npy")
    except ImportError:
        print("用户特征已保存为:")
        print(f"- JSON格式: {output_dir}/user_features_for_rumor_detection.json")

def main():
    """
    主函数 - 完整的用户特征提取和集成示例
    Main function - complete user feature extraction and integration example
    """
    print("=== 中文谣言数据集用户特征提取 - 完整示例 ===\n")
    
    # 1. 提取用户画像特征
    user_features, user_ids = create_user_profile_features_example()
    
    # 2. 与现有模型集成
    combined_features = integrate_with_existing_model_example(user_features)
    
    # 3. 保存特征供模型使用
    save_features_for_model(user_features, user_ids)
    
    # 4. 提供使用指南
    print("\n=== 使用指南 ===")
    print("1. 用户特征已按照您要求的格式提取完成")
    print("2. 特征包含: [粉丝数, 关注数, 是否认证(0/1), 账户年龄, 历史推文数]")
    print("3. 可以直接与您现有的文本特征和模型结合使用")
    print("4. 建议对特征进行标准化处理后再输入模型")
    
    print("\n代码使用示例:")
    print("""
# 加载保存的用户特征
import json
with open('/workspace/user_features_for_rumor_detection.json', 'r') as f:
    data = json.load(f)
    user_features = data['user_features']
    user_ids = data['user_ids']

# 如果安装了numpy
import numpy as np
user_features = np.array(user_features)  # [N, user_feature_dim]

# 与文本特征结合
# combined_features = np.concatenate([text_features, user_features], axis=1)
    """)

if __name__ == "__main__":
    main()