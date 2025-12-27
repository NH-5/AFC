"""
模型加载与推理模块

训练完成后，将 model.pt 文件放到 webserver/ 目录下，
然后将 USE_MOCK_MODEL 改为 False 即可使用真实模型。
"""

import torch
import random
import sys
import os
from pathlib import Path
from torchvision import transforms

# 添加 model 目录到系统路径，以便加载模型时能找到 Network 类
# 假设 model 目录在当前文件的上一级目录的 model 子目录中
sys.path.append(str(Path(__file__).parent.parent / "model"))

def get_resource_path(relative_path):
    """获取资源文件的绝对路径 (兼容 PyInstaller 打包)"""
    if hasattr(sys, '_MEIPASS'):
        # PyInstaller 解压后的临时目录
        base_path = Path(sys._MEIPASS)
    else:
        # 开发环境：项目根目录
        base_path = Path(__file__).parent.parent
    
    return base_path / relative_path

# ========== 配置 ==========
USE_MOCK_MODEL = False  # 训练完成后改为 False
MODEL_PATH = get_resource_path("best_model/model.pt")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_transform():
    """获取图片预处理变换"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

# ========== 模型加载 ==========
_model = None

def load_model():
    """加载模型（仅在首次调用时加载）"""
    global _model
    
    if USE_MOCK_MODEL:
        print("[INFO] 使用模拟模型（mock mode）")
        return None
    
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
        
        print(f"[INFO] 正在加载模型: {MODEL_PATH}")
        # weights_only=False 用于解决 PyTorch 2.6+ 默认的安全限制
        # 注意：仅加载受信任的模型文件
        _model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        _model.eval()
        print(f"[INFO] 模型加载完成，使用设备: {DEVICE}")
    
    return _model


def predict(image_tensor):
    """
    对图片进行预测
    
    Args:
        image_tensor: 预处理后的图片张量，形状为 (1, 3, 256, 256)
    
    Returns:
        dict: {
            "prediction": "AI" 或 "Real",
            "confidence": 置信度百分比,
            "probabilities": {"AI": 概率, "Real": 概率}
        }
    """
    
    if USE_MOCK_MODEL:
        # 模拟模式：返回随机结果，用于测试
        ai_prob = random.uniform(0.3, 0.95)
        real_prob = 1 - ai_prob
        
        # 随机决定预测结果
        if random.random() > 0.5:
            ai_prob, real_prob = real_prob, ai_prob
        
        prediction = "AI" if ai_prob > real_prob else "Real"
        confidence = max(ai_prob, real_prob) * 100
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "probabilities": {
                "AI": round(ai_prob, 4),
                "Real": round(real_prob, 4)
            }
        }
    
    # 真实模型推理
    model = load_model()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        
        # 假设标签顺序为 [AI, Real]（ImageFolder 按字母顺序）
        ai_prob = probabilities[0][0].item()
        real_prob = probabilities[0][1].item()
        
        prediction = "AI" if ai_prob > real_prob else "Real"
        confidence = max(ai_prob, real_prob) * 100
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "probabilities": {
                "AI": round(ai_prob, 4),
                "Real": round(real_prob, 4)
            }
        }
