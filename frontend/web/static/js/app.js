/**
 * AI Face Detection - Web Frontend
 * Minimalist White Theme
 */

class AIFaceDetectionApp {
    constructor() {
        this.serverUrl = 'http://localhost:5000/predict';
        this.currentFile = null;
        this.history = [];

        this.initElements();
        this.bindEvents();
    }

    initElements() {
        // Upload area
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');

        // Result section
        this.resultSection = document.getElementById('resultSection');
        this.previewImage = document.getElementById('previewImage');
        this.previewOverlay = document.getElementById('previewOverlay');

        // Result display
        this.resultIcon = document.getElementById('resultIcon');
        this.resultTitle = document.getElementById('resultTitle');
        this.resultSubtitle = document.getElementById('resultSubtitle');
        this.confidenceValue = document.getElementById('confidenceValue');
        this.confidenceFill = document.getElementById('confidenceFill');
        this.aiProb = document.getElementById('aiProb');
        this.aiProbBar = document.getElementById('aiProbBar');
        this.realProb = document.getElementById('realProb');
        this.realProbBar = document.getElementById('realProbBar');

        // Buttons
        this.resetBtn = document.getElementById('resetBtn');
        this.analyzeBtn = document.getElementById('analyzeBtn');

        // History
        this.historySection = document.getElementById('historySection');
        this.historyList = document.getElementById('historyList');
    }

    bindEvents() {
        // Upload area click
        this.uploadArea.addEventListener('click', () => this.fileInput.click());

        // File input change
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e.target.files[0]));

        // Drag and drop
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });

        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('dragover');
        });

        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                this.handleFileSelect(file);
            }
        });

        // Buttons
        this.analyzeBtn.addEventListener('click', () => this.analyzeImage());
        this.resetBtn.addEventListener('click', () => this.reset());
    }

    handleFileSelect(file) {
        if (!file) return;

        this.currentFile = file;
        this.showPreview(file);
    }

    showPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImage.src = e.target.result;
            this.resultSection.style.display = 'grid';
            this.uploadArea.style.display = 'none';

            // Hide result initially
            this.resetResultDisplay();

            // Scroll to result
            this.resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        };
        reader.readAsDataURL(file);
    }

    resetResultDisplay() {
        this.resultIcon.className = 'result-icon';
        this.resultIcon.querySelector('.icon-text').textContent = '--';
        this.resultTitle.textContent = '等待分析...';
        this.resultSubtitle.textContent = '上传图片后开始检测';
        this.confidenceValue.textContent = '--';
        this.confidenceFill.style.width = '0%';
        this.aiProb.textContent = '0%';
        this.aiProbBar.style.width = '0%';
        this.realProb.textContent = '0%';
        this.realProbBar.style.width = '0%';

        this.analyzeBtn.disabled = false;
        this.analyzeBtn.textContent = '开始分析';
    }

    async analyzeImage() {
        if (!this.currentFile) return;

        try {
            // Show loading
            this.previewOverlay.classList.add('active');
            this.analyzeBtn.disabled = true;
            this.analyzeBtn.textContent = '分析中...';

            // Create form data
            const formData = new FormData();
            formData.append('file', this.currentFile);

            // Send request
            const response = await fetch(this.serverUrl, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.displayResult(result);

        } catch (error) {
            console.error('Prediction error:', error);
            this.resultTitle.textContent = '分析失败';
            this.resultSubtitle.textContent = error.message || '请检查服务器是否运行';
            this.analyzeBtn.disabled = false;
            this.analyzeBtn.textContent = '重试';
        } finally {
            this.previewOverlay.classList.remove('active');
        }
    }

    displayResult(result) {
        const prediction = result.prediction;
        const confidence = result.confidence;
        const probabilities = result.probabilities;

        const aiProb = probabilities['AI'] * 100;
        const realProb = probabilities['Real'] * 100;

        // Update icon
        this.resultIcon.className = 'result-icon ' + prediction.toLowerCase();
        this.resultIcon.querySelector('.icon-text').textContent = prediction;

        // Update title
        if (prediction === 'AI') {
            this.resultTitle.textContent = 'AI 生成';
            this.resultSubtitle.textContent = '检测到这张图片可能是AI生成的';
        } else {
            this.resultTitle.textContent = '真实照片';
            this.resultSubtitle.textContent = '检测到这张图片可能是真实照片';
        }

        // Animate confidence
        this.animateValue(this.confidenceValue, '--', confidence.toFixed(1) + '%', 600);
        this.confidenceFill.className = 'confidence-fill ' + prediction.toLowerCase();
        this.confidenceFill.style.width = confidence + '%';

        // Update probabilities
        this.animateValue(this.aiProb, '0%', aiProb.toFixed(1) + '%', 500);
        this.aiProbBar.style.width = aiProb + '%';

        this.animateValue(this.realProb, '0%', realProb.toFixed(1) + '%', 500);
        this.realProbBar.style.width = realProb + '%';

        // Enable button
        this.analyzeBtn.disabled = false;
        this.analyzeBtn.textContent = '重新分析';

        // Add to history
        this.addToHistory(prediction, confidence);
    }

    animateValue(element, from, to, duration) {
        const start = performance.now();

        const update = (currentTime) => {
            const elapsed = currentTime - start;
            const progress = Math.min(elapsed / duration, 1);

            // Easing function
            const eased = 1 - Math.pow(1 - progress, 3);

            if (progress >= 1) {
                element.textContent = to;
                return;
            }

            element.textContent = to;
        };

        requestAnimationFrame(update);
    }

    addToHistory(prediction, confidence) {
        const item = {
            name: this.currentFile.name,
            time: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }),
            prediction,
            confidence: confidence.toFixed(1),
            thumbnail: this.previewImage.src
        };

        this.history.unshift(item);

        // Keep only last 12 items
        if (this.history.length > 12) {
            this.history.pop();
        }

        this.renderHistory();
    }

    renderHistory() {
        if (this.history.length === 0) {
            this.historySection.style.display = 'none';
            return;
        }

        this.historySection.style.display = 'block';

        this.historyList.innerHTML = this.history.map(item => `
            <div class="history-item">
                <img class="history-thumb" src="${item.thumbnail}" alt="${item.name}">
                <div class="history-info">
                    <div class="history-name" title="${item.name}">${item.name}</div>
                    <div class="history-meta">${item.time} · ${item.confidence}% 置信度</div>
                </div>
                <span class="history-result ${item.prediction.toLowerCase()}">${item.prediction}</span>
            </div>
        `).join('');
    }

    reset() {
        this.currentFile = null;
        this.fileInput.value = '';
        this.resultSection.style.display = 'none';
        this.uploadArea.style.display = 'block';
        this.history = [];
        this.renderHistory();

        // Reset to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new AIFaceDetectionApp();
});
