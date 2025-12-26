# -*- coding: utf-8 -*-
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
        self.root.title("AI Face Detection System")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # Server URL configuration
        self.server_url = "http://localhost:5000/predict"
        
        # Current image path
        self.current_image_path = None
        self.current_image = None
        
        # History
        self.history = []
        
        # Setup styles
        self.setup_styles()
        
        # Create UI widgets
        self.create_widgets()
        
    def setup_styles(self):
        """Setup UI styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2c3e50')
        style.configure('Info.TLabel', font=('Arial', 10), foreground='#34495e')
        style.configure('Result.TLabel', font=('Arial', 14, 'bold'))
        style.configure('AI.TLabel', foreground='#e74c3c')
        style.configure('Real.TLabel', foreground='#27ae60')
        
    def create_widgets(self):
        """Create all UI widgets"""
        # Main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights - allow resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=1)  # Left column resizable
        main_container.columnconfigure(1, weight=1)  # Right column resizable
        main_container.rowconfigure(1, weight=1)     # Main content row resizable
        
        # Title
        title_label = ttk.Label(main_container, text="AI Generated Face Detection System", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Left Panel - Image Display and Control
        left_panel = ttk.LabelFrame(main_container, text="Image Preview", padding="10")
        left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(0, weight=1)  # Image area resizable
        
        # Image Display Area
        self.image_frame = ttk.Frame(left_panel, relief=tk.SUNKEN, borderwidth=2)
        self.image_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.rowconfigure(0, weight=1)
        
        self.image_label = ttk.Label(self.image_frame, text="Please select an image", anchor=tk.CENTER)
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Button Area
        button_frame = ttk.Frame(left_panel)
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        
        self.select_btn = ttk.Button(button_frame, text="Select Image", command=self.select_image)
        self.select_btn.grid(row=0, column=0, padx=(0, 5), sticky=(tk.W, tk.E))
        
        self.predict_btn = ttk.Button(button_frame, text="Start Detection", command=self.predict_image, state=tk.DISABLED)
        self.predict_btn.grid(row=0, column=1, padx=(5, 0), sticky=(tk.W, tk.E))
        
        # Right Panel - Results and History
        right_panel = ttk.Frame(main_container)
        right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)  # History area resizable
        
        # Result Display Area
        result_frame = ttk.LabelFrame(right_panel, text="Detection Result", padding="15")
        result_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.result_label = ttk.Label(result_frame, text="Waiting...", style='Result.TLabel', anchor=tk.CENTER)
        self.result_label.pack(pady=(0, 10))
        
        self.confidence_label = ttk.Label(result_frame, text="", style='Info.TLabel', anchor=tk.CENTER)
        self.confidence_label.pack(pady=(0, 5))
        
        self.ai_prob_label = ttk.Label(result_frame, text="", style='Info.TLabel', anchor=tk.CENTER)
        self.ai_prob_label.pack(pady=(0, 5))
        
        self.real_prob_label = ttk.Label(result_frame, text="", style='Info.TLabel', anchor=tk.CENTER)
        self.real_prob_label.pack()
        
        # Progress Bar
        self.progress = ttk.Progressbar(result_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(10, 0))
        
        # History Area
        history_frame = ttk.LabelFrame(right_panel, text="History", padding="10")
        history_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        history_frame.columnconfigure(0, weight=1)
        history_frame.rowconfigure(0, weight=1)  # Treeview resizable
        
        # Create Treeview
        self.history_tree = ttk.Treeview(history_frame, columns=('Time', 'Filename', 'Result', 'Confidence'), show='headings', height=10)
        self.history_tree.heading('Time', text='Time')
        self.history_tree.heading('Filename', text='Filename')
        self.history_tree.heading('Result', text='Result')
        self.history_tree.heading('Confidence', text='Confidence')
        
        self.history_tree.column('Time', width=100)
        self.history_tree.column('Filename', width=150)
        self.history_tree.column('Result', width=80)
        self.history_tree.column('Confidence', width=80)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        self.history_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Clear History Button
        clear_btn = ttk.Button(history_frame, text="Clear History", command=self.clear_history)
        clear_btn.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        # Status Bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
    def select_image(self):
        """Select image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.predict_btn.config(state=tk.NORMAL)
            self.status_bar.config(text=f"Selected: {os.path.basename(file_path)}")
            self.reset_result()
            
    def display_image(self, image_path):
        """Display selected image"""
        try:
            # Load image
            image = Image.open(image_path)
            self.current_image = image.copy()  # Save original image
            
            # Get current display area size
            self.image_frame.update()
            frame_width = max(self.image_frame.winfo_width() - 20, 200)
            frame_height = max(self.image_frame.winfo_height() - 20, 200)
            
            # Resize image to fit display area
            image_copy = self.current_image.copy()
            image_copy.thumbnail((frame_width, frame_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image_copy)
            
            # Display image
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep reference
            
            # Bind resize event
            self.image_frame.bind('<Configure>', self.on_resize)
            
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load image: {str(e)}")
            
    def on_resize(self, event):
        """Resize image when window size changes"""
        if self.current_image:
            try:
                # Get new display area size
                frame_width = max(event.width - 20, 200)
                frame_height = max(event.height - 20, 200)
                
                # Resize image
                image_copy = self.current_image.copy()
                image_copy.thumbnail((frame_width, frame_height), Image.Resampling.LANCZOS)
                
                # Update display
                photo = ImageTk.PhotoImage(image_copy)
                self.image_label.config(image=photo)
                self.image_label.image = photo
            except:
                pass
            
    def predict_image(self):
        """Send image to server for prediction"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        # Run prediction in a new thread to avoid UI freezing
        thread = threading.Thread(target=self._predict_worker)
        thread.daemon = True
        thread.start()
        
    def _predict_worker(self):
        """Prediction worker thread"""
        try:
            # Update UI state
            self.root.after(0, self._update_predicting_state, True)
            
            # Prepare file
            files = {'file': open(self.current_image_path, 'rb')}
            
            # Send request
            response = requests.post(self.server_url, files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                self.root.after(0, self._display_result, result)
            else:
                error_msg = f"Server Error: {response.status_code}"
                self.root.after(0, messagebox.showerror, "Error", error_msg)
                
        except requests.exceptions.ConnectionError:
            self.root.after(0, messagebox.showerror, "Connection Error", 
                          "Cannot connect to server. Ensure it is running.")
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Error", f"Prediction Failed: {str(e)}")
        finally:
            self.root.after(0, self._update_predicting_state, False)
            
    def _update_predicting_state(self, is_predicting):
        """Update prediction state"""
        if is_predicting:
            self.predict_btn.config(state=tk.DISABLED)
            self.select_btn.config(state=tk.DISABLED)
            self.progress.start(10)
            self.status_bar.config(text="Detecting...")
        else:
            self.predict_btn.config(state=tk.NORMAL)
            self.select_btn.config(state=tk.NORMAL)
            self.progress.stop()
            self.status_bar.config(text="Detection Complete")
            
    def _display_result(self, result):
        """Display prediction result"""
        prediction = result.get('prediction', 'Unknown')
        confidence = result.get('confidence', 0)
        probabilities = result.get('probabilities', {})
        
        ai_prob = probabilities.get('AI', 0) * 100
        real_prob = probabilities.get('Real', 0) * 100
        
        # Update result display
        if prediction == 'AI':
            self.result_label.config(text="AI Generated", style='AI.TLabel')
        else:
            self.result_label.config(text="Real Photo", style='Real.TLabel')
        
        self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")
        self.ai_prob_label.config(text=f"AI Probability: {ai_prob:.2f}%")
        self.real_prob_label.config(text=f"Real Probability: {real_prob:.2f}%")
        
        # Add to history
        self.add_to_history(prediction, confidence)
        
    def add_to_history(self, prediction, confidence):
        """Add detection record to history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        filename = os.path.basename(self.current_image_path)
        
        self.history_tree.insert('', 0, values=(
            timestamp,
            filename,
            prediction,
            f"{confidence:.2f}%"
        ))
        
    def clear_history(self):
        """Clear history records"""
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        self.status_bar.config(text="History Cleared")
        
    def reset_result(self):
        """Reset result display"""
        self.result_label.config(text="Waiting...", style='Result.TLabel')
        self.confidence_label.config(text="")
        self.ai_prob_label.config(text="")
        self.real_prob_label.config(text="")

def main():
    root = tk.Tk()
    app = AIFaceDetectionUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
