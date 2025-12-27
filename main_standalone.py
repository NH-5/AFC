import sys
import os
from PIL import Image
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread, pyqtSignal

# Add project root to sys.path if not present
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from frontend.qtui.main import MainWindow, PredictWorker
from backend.model_loader import predict, get_transform

class LocalPredictWorker(QThread):
    """
    Local prediction worker that runs inference directly
    instead of calling a server.
    """
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, image_path: str):
        super().__init__()
        self.image_path = image_path

    def run(self):
        try:
            # Load and preprocess image
            image = Image.open(self.image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            transform = get_transform()
            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

            # Run inference
            result = predict(image_tensor)
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))

# Patch the MainWindow to use LocalPredictWorker
class StandaloneMainWindow(MainWindow):
    def analyze_image(self):
        """Override to use LocalPredictWorker"""
        if not self.current_image_path:
            return

        self.analyze_btn.setEnabled(False)
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.update_status("Analyzing (Local Inference)...")

        # Use LocalPredictWorker instead of the default one
        self.worker = LocalPredictWorker(self.current_image_path)
        self.worker.finished.connect(self.on_analysis_complete)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AFC - AI Face Detection (Standalone)")

    window = StandaloneMainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
