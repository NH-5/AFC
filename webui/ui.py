import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import requests
import json
import os
from datetime import datetime
import threading

class AIFaceDetectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI人脸检测系统")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # 配置服务器地址
        self.server_url = "http://localhost:5000/predict"
        
        # 当前图片路径
        self.current_image_path = None
        self.current_image = None
        
        # 历史记录
        self.history = []
        
        # 设置样式
        self.setup_styles()
        
        # 创建UI组件
        self.create_widgets()
        
    def setup_styles(self):
        """设置UI样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 配置颜色
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2c3e50')
        style.configure('Info.TLabel', font=('Arial', 10), foreground='#34495e')
        style.configure('Result.TLabel', font=('Arial', 14, 'bold'))
        style.configure('AI.TLabel', foreground='#e74c3c')
        style.configure('Real.TLabel', foreground='#27ae60')
        
    def create_widgets(self):
        """创建所有UI组件"""
        # 主容器
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重 - 使整个界面可以拉伸
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=1)  # 左侧列可拉伸
        main_container.columnconfigure(1, weight=1)  # 右侧列可拉伸
        main_container.rowconfigure(1, weight=1)     # 主内容行可拉伸
        
        # 标题
        title_label = ttk.Label(main_container, text="🤖 AI生成人脸检测系统", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # 左侧面板 - 图片显示和控制
        left_panel = ttk.LabelFrame(main_container, text="图片预览", padding="10")
        left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(0, weight=1)  # 图片区域可拉伸
        
        # 图片显示区域
        self.image_frame = ttk.Frame(left_panel, relief=tk.SUNKEN, borderwidth=2)
        self.image_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.rowconfigure(0, weight=1)
        
        self.image_label = ttk.Label(self.image_frame, text="请选择图片", anchor=tk.CENTER)
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 按钮区域
        button_frame = ttk.Frame(left_panel)
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        
        self.select_btn = ttk.Button(button_frame, text="📁 选择图片", command=self.select_image)
        self.select_btn.grid(row=0, column=0, padx=(0, 5), sticky=(tk.W, tk.E))
        
        self.predict_btn = ttk.Button(button_frame, text="🔍 开始检测", command=self.predict_image, state=tk.DISABLED)
        self.predict_btn.grid(row=0, column=1, padx=(5, 0), sticky=(tk.W, tk.E))
        
        # 右侧面板 - 结果和历史
        right_panel = ttk.Frame(main_container)
        right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)  # 历史记录区域可拉伸
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(right_panel, text="检测结果", padding="15")
        result_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.result_label = ttk.Label(result_frame, text="等待检测...", style='Result.TLabel', anchor=tk.CENTER)
        self.result_label.pack(pady=(0, 10))
        
        self.confidence_label = ttk.Label(result_frame, text="", style='Info.TLabel', anchor=tk.CENTER)
        self.confidence_label.pack(pady=(0, 5))
        
        self.ai_prob_label = ttk.Label(result_frame, text="", style='Info.TLabel', anchor=tk.CENTER)
        self.ai_prob_label.pack(pady=(0, 5))
        
        self.real_prob_label = ttk.Label(result_frame, text="", style='Info.TLabel', anchor=tk.CENTER)
        self.real_prob_label.pack()
        
        # 进度条
        self.progress = ttk.Progressbar(result_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(10, 0))
        
        # 历史记录区域
        history_frame = ttk.LabelFrame(right_panel, text="检测历史", padding="10")
        history_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        history_frame.columnconfigure(0, weight=1)
        history_frame.rowconfigure(0, weight=1)  # Treeview可拉伸
        
        # 创建Treeview
        self.history_tree = ttk.Treeview(history_frame, columns=('时间', '文件名', '结果', '置信度'), show='headings', height=10)
        self.history_tree.heading('时间', text='时间')
        self.history_tree.heading('文件名', text='文件名')
        self.history_tree.heading('结果', text='结果')
        self.history_tree.heading('置信度', text='置信度')
        
        self.history_tree.column('时间', width=100)
        self.history_tree.column('文件名', width=150)
        self.history_tree.column('结果', width=80)
        self.history_tree.column('置信度', width=80)
        
        # 滚动条
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        self.history_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 清除历史按钮
        clear_btn = ttk.Button(history_frame, text="清除历史", command=self.clear_history)
        clear_btn.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        # 状态栏
        self.status_bar = ttk.Label(self.root, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
    def select_image(self):
        """选择图片文件"""
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.predict_btn.config(state=tk.NORMAL)
            self.status_bar.config(text=f"已选择: {os.path.basename(file_path)}")
            self.reset_result()
            
    def display_image(self, image_path):
        """显示选择的图片"""
        try:
            # 加载图片
            image = Image.open(image_path)
            self.current_image = image.copy()  # 保存原始图片
            
            # 获取当前显示区域的大小
            self.image_frame.update()
            frame_width = max(self.image_frame.winfo_width() - 20, 200)
            frame_height = max(self.image_frame.winfo_height() - 20, 200)
            
            # 调整图片大小以适应显示区域
            image_copy = self.current_image.copy()
            image_copy.thumbnail((frame_width, frame_height), Image.Resampling.LANCZOS)
            
            # 转换为PhotoImage
            photo = ImageTk.PhotoImage(image_copy)
            
            # 显示图片
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # 保持引用
            
            # 绑定窗口大小改变事件
            self.image_frame.bind('<Configure>', self.on_resize)
            
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片: {str(e)}")
            
    def on_resize(self, event):
        """窗口大小改变时重新调整图片"""
        if self.current_image:
            try:
                # 获取新的显示区域大小
                frame_width = max(event.width - 20, 200)
                frame_height = max(event.height - 20, 200)
                
                # 调整图片大小
                image_copy = self.current_image.copy()
                image_copy.thumbnail((frame_width, frame_height), Image.Resampling.LANCZOS)
                
                # 更新显示
                photo = ImageTk.PhotoImage(image_copy)
                self.image_label.config(image=photo)
                self.image_label.image = photo
            except:
                pass
            
    def predict_image(self):
        """发送图片到服务器进行预测"""
        if not self.current_image_path:
            messagebox.showwarning("警告", "请先选择图片")
            return
        
        # 在新线程中执行预测,避免UI冻结
        thread = threading.Thread(target=self._predict_worker)
        thread.daemon = True
        thread.start()
        
    def _predict_worker(self):
        """预测工作线程"""
        try:
            # 更新UI状态
            self.root.after(0, self._update_predicting_state, True)
            
            # 准备文件
            files = {'file': open(self.current_image_path, 'rb')}
            
            # 发送请求
            response = requests.post(self.server_url, files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                self.root.after(0, self._display_result, result)
            else:
                error_msg = f"服务器错误: {response.status_code}"
                self.root.after(0, messagebox.showerror, "错误", error_msg)
                
        except requests.exceptions.ConnectionError:
            self.root.after(0, messagebox.showerror, "连接错误", 
                          "无法连接到服务器,请确保服务器正在运行")
        except Exception as e:
            self.root.after(0, messagebox.showerror, "错误", f"预测失败: {str(e)}")
        finally:
            self.root.after(0, self._update_predicting_state, False)
            
    def _update_predicting_state(self, is_predicting):
        """更新预测状态"""
        if is_predicting:
            self.predict_btn.config(state=tk.DISABLED)
            self.select_btn.config(state=tk.DISABLED)
            self.progress.start(10)
            self.status_bar.config(text="正在检测...")
        else:
            self.predict_btn.config(state=tk.NORMAL)
            self.select_btn.config(state=tk.NORMAL)
            self.progress.stop()
            self.status_bar.config(text="检测完成")
            
    def _display_result(self, result):
        """显示预测结果"""
        prediction = result.get('prediction', 'Unknown')
        confidence = result.get('confidence', 0)
        probabilities = result.get('probabilities', {})
        
        ai_prob = probabilities.get('AI', 0) * 100
        real_prob = probabilities.get('Real', 0) * 100
        
        # 更新结果显示
        if prediction == 'AI':
            self.result_label.config(text="⚠️ AI生成", style='AI.TLabel')
        else:
            self.result_label.config(text="✓ 真实照片", style='Real.TLabel')
        
        self.confidence_label.config(text=f"置信度: {confidence:.2f}%")
        self.ai_prob_label.config(text=f"AI生成概率: {ai_prob:.2f}%")
        self.real_prob_label.config(text=f"真实照片概率: {real_prob:.2f}%")
        
        # 添加到历史记录
        self.add_to_history(prediction, confidence)
        
    def add_to_history(self, prediction, confidence):
        """添加检测记录到历史"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        filename = os.path.basename(self.current_image_path)
        
        self.history_tree.insert('', 0, values=(
            timestamp,
            filename,
            prediction,
            f"{confidence:.2f}%"
        ))
        
    def clear_history(self):
        """清除历史记录"""
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        self.status_bar.config(text="历史记录已清除")
        
    def reset_result(self):
        """重置结果显示"""
        self.result_label.config(text="等待检测...", style='Result.TLabel')
        self.confidence_label.config(text="")
        self.ai_prob_label.config(text="")
        self.real_prob_label.config(text="")

def main():
    root = tk.Tk()
    app = AIFaceDetectionUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
