"""
AFC 后端服务器

启动方式：python app.py
API 端点：POST /predict
网页前端: http://localhost:5000/
"""

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from torchvision import transforms
import io
from pathlib import Path

from model_loader import predict, load_model

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
WEB_FRONTEND_PATH = PROJECT_ROOT / 'frontend' / 'web'

app = Flask(
    __name__,
    template_folder=WEB_FRONTEND_PATH,
    static_folder=WEB_FRONTEND_PATH / 'static'
)

# 图片预处理（与训练时保持一致）
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


@app.route('/predict', methods=['POST'])
def predict_image():
    """
    预测图片是否为AI生成
    
    请求：multipart/form-data，字段 file 为图片文件
    响应：JSON {prediction, confidence, probabilities}
    """
    
    # 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify({"error": "没有上传文件"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "文件名为空"}), 400
    
    try:
        # 读取图片
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # 转换为 RGB（处理 RGBA 或灰度图）
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 预处理
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # 添加 batch 维度
        
        # 预测
        result = predict(image_tensor)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"处理图片时出错: {str(e)}"}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({"status": "ok"})


@app.route('/')
def index():
    """网页前端首页"""
    return send_from_directory(WEB_FRONTEND_PATH / 'templates', 'index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    """提供静态文件服务"""
    return send_from_directory(WEB_FRONTEND_PATH / 'static', filename)

if __name__ == '__main__':
    # 预加载模型
    print("=" * 50)
    print("AFC 后端服务器启动中...")
    print("=" * 50)
    load_model()
    
    # 启动服务器
    print("\n服务器地址: http://localhost:5000")
    print("API 端点: POST /predict")
    print("健康检查: GET /health")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
