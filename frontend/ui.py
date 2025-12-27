# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import requests
import os
from datetime import datetime
import threading


class iOSButton(tk.Canvas):
    """iOS风格按钮"""

    def __init__(self, parent, text, command, width=160, height=44,
                 bg_color="#007AFF", hover_color="#0056B3",
                 text_color="white", font=("-apple-system", 13, "bold"), **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=parent.cget('bg'), highlightthickness=0, **kwargs)

        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.font = font
        self.text = text
        self.radius = 10
        self.current_color = bg_color

        self.draw_button(bg_color)
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
        self.bind("<ButtonRelease-1>", self.on_release)

    def draw_button(self, color):
        self.delete("all")
        w, h, r = self.winfo_reqwidth(), self.winfo_reqheight(), self.radius

        # Rounded rectangle with subtle shadow effect
        self.create_arc(0, 0, r * 2, r * 2, start=90, extent=90, fill=color, outline=color)
        self.create_arc(w - r * 2, 0, w, r * 2, start=0, extent=90, fill=color, outline=color)
        self.create_arc(0, h - r * 2, r * 2, h, start=180, extent=90, fill=color, outline=color)
        self.create_arc(w - r * 2, h - r * 2, w, h, start=270, extent=90, fill=color, outline=color)
        self.create_rectangle(r, 0, w - r, h, fill=color, outline=color)
        self.create_rectangle(0, r, w, h - r, fill=color, outline=color)

        # Highlight at top
        self.create_rectangle(r, 2, w - r, 4, fill="white", outline="white")
        self.create_arc(r, 0, r * 2, 4, start=180, extent=90, fill="white", outline="white")
        self.create_arc(w - r * 2, 0, w, 4, start=270, extent=90, fill="white", outline="white")

        self.create_text(w / 2, h / 2, text=self.text, fill=self.text_color, font=self.font)

    def on_enter(self, event):
        self.draw_button(self.hover_color)
        self.config(cursor="hand2")

    def on_leave(self, event):
        self.draw_button(self.bg_color)
        self.config(cursor="")

    def on_click(self, event):
        self.draw_button(self.hover_color)
        self.config(relief=tk.SUNKEN)

    def on_release(self, event):
        self.draw_button(self.hover_color)
        self.config(relief=tk.FLAT)
        if self.command:
            self.command()


class iOSCard(tk.Frame):
    """iOS风格卡片容器"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg="#FFFFFF", **kwargs)

        # Main content frame
        self.content = tk.Frame(self, bg="#FFFFFF", relief=tk.FLAT)
        self.content.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        # Subtle border
        self.config(bg="#E5E5E5", bd=0, relief=tk.FLAT)


class iOSProgressBar(tk.Frame):
    """iOS风格进度条"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg="#FFFFFF", **kwargs)

        self.bg_bar = tk.Frame(self, bg="#E9E9E9", height=4, relief=tk.FLAT)
        self.bg_bar.pack(fill=tk.X)
        self.bg_bar.pack_propagate(False)

        self.progress_bar = tk.Frame(self.bg_bar, bg="#007AFF", width=0, height=4, relief=tk.FLAT)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.Y)

    def set_progress(self, value):
        max_width = self.bg_bar.winfo_reqwidth()
        width = (value / 100) * max_width
        self.progress_bar.config(width=int(width))

    def start(self):
        self._animate()

    def _animate(self):
        if self.progress_bar.cget('width') == '' or int(self.progress_bar.cget('width')) == 0:
            return

    def stop(self):
        pass


class iOSToggle(tk.Frame):
    """iOS风格开关"""

    def __init__(self, parent, text, color="#007AFF", **kwargs):
        super().__init__(parent, bg="#FFFFFF", **kwargs)

        self.color = color
        self.is_on = False

        # Track
        self.track = tk.Frame(self, bg="#E9E9E9", width=51, height=31, relief=tk.FLAT, bd=0)
        self.track.pack(side=tk.LEFT)
        self.track.pack_propagate(False)

        # Thumb
        self.thumb = tk.Frame(self.track, bg="#FFFFFF", width=27, height=27,
                              relief=tk.FLAT, bd=0)
        self.thumb.place(relx=2, rely=2)

        # Label
        self.label = tk.Label(self, text=text, font=("-apple-system", 14),
                              bg="#FFFFFF", fg="#000000")
        self.label.pack(side=tk.LEFT, padx=10)


class AIFaceDetectionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Face Detection")
        self.root.geometry("1100x750")
        self.root.minsize(900, 650)
        self.root.resizable(True, True)

        # iOS color palette
        self.colors = {
            "bg": "#F2F2F7",
            "card_bg": "#FFFFFF",
            "primary": "#007AFF",
            "primary_hover": "#0056B3",
            "success": "#34C759",
            "danger": "#FF3B30",
            "warning": "#FF9500",
            "text_primary": "#000000",
            "text_secondary": "#8E8E93",
            "separator": "#C6C6C8",
            "input_bg": "#E9E9E9",
            "border": "#E5E5E5"
        }

        # Fonts
        self.fonts = {
            "title": ("-apple-system", 28, "bold"),
            "subtitle": ("-apple-system", 13, "regular"),
            "heading": ("-apple-system", 17, "bold"),
            "body": ("-apple-system", 15, "regular"),
            "caption": ("-apple-system", 12, "regular"),
            "button": ("-apple-system", 14, "bold")
        }

        self.server_url = "http://localhost:5000/predict"
        self.current_image_path = None
        self.current_image = None

        self.root.configure(bg=self.colors["bg"])
        self.create_widgets()

    def create_widgets(self):
        """Create iOS-style UI"""
        # Main scrollable container
        self.main_scroll = tk.Frame(self.root, bg=self.colors["bg"])
        self.main_scroll.pack(fill=tk.BOTH, expand=True)

        # Header section
        self.create_header(self.main_scroll)

        # Content area
        content = tk.Frame(self.main_scroll, bg=self.colors["bg"])
        content.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        # Two column layout
        left_col = tk.Frame(content, bg=self.colors["bg"])
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        right_col = tk.Frame(content, bg=self.colors["bg"])
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))

        # Image panel (left)
        self.image_card = self.create_image_card(left_col)
        self.image_card.pack(fill=tk.BOTH, expand=True)

        # Result panel (right)
        self.result_card = self.create_result_card(right_col)
        self.result_card.pack(fill=tk.BOTH)

        # Status bar
        self.create_status_bar(self.main_scroll)

    def create_header(self, parent):
        """Create iOS-style header"""
        header = tk.Frame(parent, bg=self.colors["bg"])
        header.pack(fill=tk.X, padx=20, pady=(20, 8))

        # Large title
        title = tk.Label(header, text="Face Detection",
                         font=self.fonts["title"], bg=self.colors["bg"],
                         fg=self.colors["text_primary"])
        title.pack(anchor=tk.W)

        # Subtitle
        subtitle = tk.Label(header,
                          text="AI-powered image analysis",
                          font=self.fonts["subtitle"], bg=self.colors["bg"],
                          fg=self.colors["text_secondary"])
        subtitle.pack(anchor=tk.W, pady=(4, 0))

    def create_image_card(self, parent):
        """Create image preview card"""
        card = iOSCard(parent)
        card.pack(fill=tk.BOTH, expand=True)

        content = card.content

        # Section header
        header = tk.Frame(content, bg=self.colors["card_bg"])
        header.pack(fill=tk.X, padx=16, pady=(12, 8))

        header_label = tk.Label(header, text="Image",
                                font=self.fonts["heading"], bg=self.colors["card_bg"],
                                fg=self.colors["text_primary"])
        header_label.pack(anchor=tk.W)

        # Image display area
        self.image_frame = tk.Frame(content, bg="#F9F9F9",
                                    relief=tk.FLAT, bd=0)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 12))

        image_inner = tk.Frame(self.image_frame, bg="#F9F9F9")
        image_inner.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.image_label = tk.Label(image_inner,
                                    text="No Image Selected\n\nSelect an image to begin",
                                    font=self.fonts["body"], bg="#F9F9F9",
                                    fg=self.colors["text_secondary"],
                                    justify=tk.CENTER)
        self.image_label.pack(padx=40, pady=40)

        # Button row
        btn_row = tk.Frame(content, bg=self.colors["card_bg"])
        btn_row.pack(fill=tk.X, padx=16, pady=(0, 12))

        self.select_btn = iOSButton(
            btn_row, "Select Image",
            command=self.select_image,
            width=150, height=44,
            bg_color=self.colors["primary"],
            font=self.fonts["button"]
        )
        self.select_btn.pack(side=tk.LEFT)

        self.predict_btn = iOSButton(
            btn_row, "Analyze",
            command=self.predict_image,
            width=150, height=44,
            bg_color=self.colors["success"],
            font=self.fonts["button"]
        )
        self.predict_btn.pack(side=tk.LEFT, padx=12)

        # Progress bar
        self.progress = iOSProgressBar(content)
        self.progress.pack(fill=tk.X, padx=16, pady=(0, 12))

        return card

    def create_result_card(self, parent):
        """Create result display card"""
        card = iOSCard(parent)
        card.pack(fill=tk.BOTH)

        content = card.content

        # Section header
        header = tk.Frame(content, bg=self.colors["card_bg"])
        header.pack(fill=tk.X, padx=16, pady=(12, 8))

        header_label = tk.Label(header, text="Results",
                                font=self.fonts["heading"], bg=self.colors["card_bg"],
                                fg=self.colors["text_primary"])
        header_label.pack(anchor=tk.W)

        # Result icon container
        icon_container = tk.Frame(content, bg=self.colors["card_bg"])
        icon_container.pack(fill=tk.X, padx=16, pady=(20, 0))

        self.result_icon = tk.Label(icon_container, text="--",
                                    font=("SF Pro Display", 48),
                                    bg=self.colors["card_bg"],
                                    fg=self.colors["text_secondary"])
        self.result_icon.pack(pady=(0, 8))

        # Result text
        self.result_label = tk.Label(content, text="Waiting for image...",
                                     font=self.fonts["body"], bg=self.colors["card_bg"],
                                     fg=self.colors["text_secondary"])
        self.result_label.pack(pady=8)

        # Confidence section
        conf_frame = tk.Frame(content, bg=self.colors["card_bg"])
        conf_frame.pack(fill=tk.X, padx=16, pady=(24, 0))

        conf_title = tk.Label(conf_frame, text="Confidence",
                              font=self.fonts["caption"], bg=self.colors["card_bg"],
                              fg=self.colors["text_secondary"])
        conf_title.pack(anchor=tk.W)

        # Confidence bar
        self.conf_bar_bg = tk.Frame(content, bg=self.colors["input_bg"],
                                    height=6, relief=tk.FLAT)
        self.conf_bar_bg.pack(fill=tk.X, padx=16, pady=(8, 0))
        self.conf_bar_bg.pack_propagate(False)

        self.conf_bar = tk.Frame(self.conf_bar_bg, bg=self.colors["primary"],
                                 width=0, height=6, relief=tk.FLAT)
        self.conf_bar.pack(side=tk.LEFT, fill=tk.Y)

        self.conf_label = tk.Label(content, text="--",
                                   font=self.fonts["caption"], bg=self.colors["card_bg"],
                                   fg=self.colors["text_primary"])
        self.conf_label.pack(anchor=tk.E, padx=16, pady=(6, 0))

        # Probability breakdown
        prob_frame = tk.Frame(content, bg=self.colors["card_bg"])
        prob_frame.pack(fill=tk.X, padx=16, pady=(24, 0))

        # AI Probability
        self.create_prob_row(prob_frame, "AI Generated", 0)
        # Real Probability
        self.create_prob_row(prob_frame, "Real Photo", 1)

        # History section
        history_frame = tk.Frame(content, bg=self.colors["card_bg"])
        history_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(24, 16))

        history_title = tk.Label(history_frame, text="Recent",
                                 font=self.fonts["caption"], bg=self.colors["card_bg"],
                                 fg=self.colors["text_secondary"])
        history_title.pack(anchor=tk.W)

        # Treeview
        self.history_frame = tk.Frame(history_frame, bg=self.colors["card_bg"])
        self.history_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        style = ttk.Style()
        style.configure("iOS.Treeview",
                        background="#FFFFFF",
                        foreground=self.colors["text_primary"],
                        fieldbackground="#FFFFFF",
                        font=self.fonts["caption"],
                        rowheight=36,
                        borderwidth=0,
                        relief=tk.FLAT)
        style.configure("iOS.Treeview.Heading",
                        font=self.fonts["caption"],
                        background=self.colors["bg"],
                        foreground=self.colors["text_secondary"],
                        borderwidth=0,
                        relief=tk.FLAT)
        style.map("iOS.Treeview",
                  background=[('selected', self.colors["primary"])])

        self.history_tree = ttk.Treeview(
            self.history_frame,
            columns=('Time', 'Name', 'Result', 'Conf'),
            show='headings',
            style="iOS.Treeview",
            height=5
        )

        self.history_tree.heading('Time', text='Time')
        self.history_tree.heading('Name', text='Name')
        self.history_tree.heading('Result', text='Result')
        self.history_tree.heading('Conf', text='Conf')

        self.history_tree.column('Time', width=60, anchor=tk.CENTER)
        self.history_tree.column('Name', width=100, anchor=tk.CENTER)
        self.history_tree.column('Result', width=80, anchor=tk.CENTER)
        self.history_tree.column('Conf', width=60, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(self.history_frame, orient=tk.VERTICAL,
                                  command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)

        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Clear button
        clear_btn = iOSButton(
            content, "Clear History",
            command=self.clear_history,
            width=140, height=36,
            bg_color="#8E8E93",
            font=("-apple-system", 12, "bold")
        )
        clear_btn.pack(pady=(12, 16))

        return card

    def create_prob_row(self, parent, text, row):
        """Create probability display row"""
        frame = tk.Frame(parent, bg=self.colors["card_bg"])
        frame.grid(row=row, column=0, sticky=tk.W + tk.E, pady=6)

        label = tk.Label(frame, text=text, font=self.fonts["body"],
                         bg=self.colors["card_bg"], fg=self.colors["text_primary"])
        label.pack(side=tk.LEFT)

        value_frame = tk.Frame(frame, bg=self.colors["card_bg"])
        value_frame.pack(side=tk.RIGHT)

        prob_bg = tk.Frame(value_frame, bg=self.colors["input_bg"],
                           width=100, height=6, relief=tk.FLAT)
        prob_bg.pack(side=tk.RIGHT)
        prob_bg.pack_propagate(False)

        prob_fill = tk.Frame(prob_bg, bg=self.colors["danger"] if row == 0 else self.colors["success"],
                             width=0, height=6, relief=tk.FLAT)

        percentage = tk.Label(value_frame, text="0%", font=self.fonts["caption"],
                              bg=self.colors["card_bg"],
                              fg=self.colors["text_primary"], width=6, anchor=tk.E)
        percentage.pack(side=tk.RIGHT, padx=(8, 0))

        if row == 0:
            self.ai_prob_bar = prob_fill
            self.ai_prob_pct = percentage
        else:
            self.real_prob_bar = prob_fill
            self.real_prob_pct = percentage

        prob_fill.pack(side=tk.LEFT, fill=tk.Y)

    def create_status_bar(self, parent):
        """Create iOS-style status bar"""
        self.status_bar = tk.Frame(parent, bg=self.colors["primary"], height=24)
        self.status_bar.pack(fill=tk.X)

        self.status_label = tk.Label(self.status_bar, text="Ready",
                                     font=self.fonts["caption"],
                                     bg=self.colors["primary"], fg="white")
        self.status_label.pack(side=tk.LEFT, padx=12, pady=4)

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
            self.update_status(f"Selected: {os.path.basename(file_path)}")
            self.reset_result()

    def display_image(self, image_path):
        """Display selected image"""
        try:
            image = Image.open(image_path)
            self.current_image = image.copy()

            self.image_frame.update()
            frame_width = max(self.image_frame.winfo_width() - 20, 250)
            frame_height = max(self.image_frame.winfo_height() - 20, 250)

            image_copy = self.current_image.copy()
            image_copy.thumbnail((frame_width, frame_height), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(image_copy)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo

        except Exception as e:
            messagebox.showerror("Error", f"Cannot load image: {str(e)}")

    def predict_image(self):
        """Send image to server for prediction"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first")
            return

        thread = threading.Thread(target=self._predict_worker)
        thread.daemon = True
        thread.start()

    def _predict_worker(self):
        """Prediction worker thread"""
        try:
            self.root.after(0, self._update_predicting_state, True)

            files = {'file': open(self.current_image_path, 'rb')}
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
            self.progress.set_progress(0)
            self.update_status("Analyzing...", self.colors["warning"])

            # Animate progress
            for i in range(0, 101, 2):
                if self.select_btn.cget('state') == tk.NORMAL:
                    break
                self.progress.set_progress(i)
                self.progress.update()
                self.progress.after(20)

    def _display_result(self, result):
        """Display prediction result"""
        prediction = result.get('prediction', 'Unknown')
        confidence = result.get('confidence', 0)
        probabilities = result.get('probabilities', {})

        ai_prob = probabilities.get('AI', 0) * 100
        real_prob = probabilities.get('Real', 0) * 100

        # Determine icon and colors
        if prediction == 'AI':
            self.result_icon.config(text="AI", fg=self.colors["danger"])
            self.result_label.config(text="AI Generated", fg=self.colors["danger"])
            self.conf_bar.config(bg=self.colors["danger"])
            result_text = "AI"
        else:
            self.result_icon.config(text="REAL", fg=self.colors["success"])
            self.result_label.config(text="Real Photo", fg=self.colors["success"])
            self.conf_bar.config(bg=self.colors["success"])
            result_text = "Real"

        # Animate confidence bar
        self.conf_bar.config(width=0)
        self.conf_bar.update()
        target_width = (confidence / 100) * self.conf_bar_bg.winfo_reqwidth()
        for i in range(0, int(target_width) + 1, 3):
            self.conf_bar.config(width=i)
            self.conf_bar.update()
            self.conf_bar.after(3)

        self.conf_label.config(text=f"{confidence:.1f}%")

        # Update probability bars
        self.update_prob_bar(self.ai_prob_bar, self.ai_prob_pct, ai_prob)
        self.update_prob_bar(self.real_prob_bar, self.real_prob_pct, real_prob)

        # Add to history
        self.add_to_history(result_text, confidence)

        self.update_status("Analysis complete", self.colors["success"])

    def update_prob_bar(self, bar, label, value):
        """Update probability bar animation"""
        bar_width = (value / 100) * 100
        label.config(text=f"{value:.1f}%")
        bar.update()
        for i in range(0, int(bar_width) + 1, 4):
            bar.config(width=i)
            bar.update()
            bar.after(2)

    def add_to_history(self, prediction, confidence):
        """Add detection record to history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        filename = os.path.basename(self.current_image_path)

        self.history_tree.insert('', 0, values=(
            timestamp,
            filename[:12] + ".." if len(filename) > 14 else filename,
            prediction,
            f"{confidence:.1f}%"
        ))

        if len(self.history_tree.get_children()) > 30:
            self.history_tree.delete(self.history_tree.get_children()[-1])

    def clear_history(self):
        """Clear history records"""
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        self.update_status("History cleared", self.colors["text_secondary"])

    def update_status(self, text, color=None):
        """Update status bar"""
        bg = color if color else self.colors["primary"]
        self.status_bar.config(bg=bg)
        self.status_label.config(bg=bg, text=text)

    def reset_result(self):
        """Reset result display"""
        self.result_icon.config(text="--", fg=self.colors["text_secondary"])
        self.result_label.config(text="Waiting for image...", fg=self.colors["text_secondary"])
        self.conf_bar.config(width=0, bg=self.colors["primary"])
        self.conf_label.config(text="--")
        if hasattr(self, 'ai_prob_bar'):
            self.update_prob_bar(self.ai_prob_bar, self.ai_prob_pct, 0)
            self.update_prob_bar(self.real_prob_bar, self.real_prob_pct, 0)


def main():
    root = tk.Tk()

    # DPI awareness for sharper text
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    app = AIFaceDetectionUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
