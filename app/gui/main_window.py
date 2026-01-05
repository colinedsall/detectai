"""
Main window for DetectAI Training Manager.
"""
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QTextEdit, QProgressBar, QLabel, QPushButton, QComboBox,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, QSplitter, QPlainTextEdit,
    QMessageBox, QFileDialog, QTextBrowser, QScrollArea
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import os
import yaml

from .workers import DataCollectionWorker, TrainingWorker

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DetectAI Training Manager")
        self.setMinimumSize(1200, 700)
        
        # Workers
        self.collection_worker = None
        self.training_worker = None
        
        self._setup_ui()
        self._load_config()
    
    def _setup_ui(self):
        """Set up the main UI."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Main horizontal splitter (tabs on left, system monitor on right)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: tabs and terminal
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create splitter for tabs and terminal
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Tab widget
        self.tabs = QTabWidget()
        self._create_config_tab()
        self._create_collection_tab()
        self._create_training_tab()
        self._create_detection_tab()
        self._create_stats_tab()
        splitter.addWidget(self.tabs)
        
        # Terminal output
        terminal_group = QGroupBox("Terminal Output")
        terminal_layout = QVBoxLayout(terminal_group)
        self.terminal = QPlainTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setFont(QFont("Menlo", 11))
        self.terminal.setMaximumBlockCount(1000)
        terminal_layout.addWidget(self.terminal)
        splitter.addWidget(terminal_group)
        
        splitter.setSizes([500, 200])
        left_layout.addWidget(splitter)
        
        main_splitter.addWidget(left_widget)
        
        # Right side: System Monitor with Quit button
        from app.gui.system_monitor import SystemMonitorWidget
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        self.system_monitor = SystemMonitorWidget()
        right_layout.addWidget(self.system_monitor)
        
        # Stop button - cancels any running workers
        self.stop_btn = QPushButton("Stop Current Action")
        self.stop_btn.clicked.connect(self._stop_all_workers)
        self.stop_btn.setStyleSheet("background-color: #DAA520; color: black; font-weight: bold;")
        right_layout.addWidget(self.stop_btn)
        
        # Quit button at bottom of right sidebar
        quit_btn = QPushButton("Quit")
        quit_btn.clicked.connect(self.close)
        quit_btn.setStyleSheet("background-color: #8B0000; color: white; font-weight: bold;")
        right_layout.addWidget(quit_btn)
        
        main_splitter.addWidget(right_widget)
        
        # Set sizes (80% left, 20% right) and fix min width for sidebar
        right_widget.setMinimumWidth(300)
        right_widget.setMaximumWidth(400)
        main_splitter.setSizes([900, 200])
        
        layout.addWidget(main_splitter)
        
        # Progress bar
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_bar, stretch=1)
        progress_layout.addWidget(self.status_label)
        layout.addLayout(progress_layout)
    
    def _create_config_tab(self):
        """Create the configuration editor tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Config editor
        self.config_editor = QTextEdit()
        self.config_editor.setFont(QFont("Menlo", 12))
        
        # Apply syntax highlighting
        try:
            from .yaml_highlighter import YamlHighlighter
            self.highlighter = YamlHighlighter(self.config_editor.document())
        except ImportError:
            print("Could not import YamlHighlighter")
            
        layout.addWidget(self.config_editor)
        
        # Buttons
        btn_layout = QHBoxLayout()
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._load_config)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save_config)
        btn_layout.addStretch()
        btn_layout.addWidget(reload_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)
        
        self.tabs.addTab(tab, "Config")
    
    def _create_collection_tab(self):
        """Create the data collection tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Settings
        settings_group = QGroupBox("Collection Settings")
        settings_layout = QFormLayout(settings_group)
        
        self.human_samples_spin = QSpinBox()
        self.human_samples_spin.setRange(5, 100)
        self.human_samples_spin.setValue(20)
        settings_layout.addRow("Human articles per site:", self.human_samples_spin)
        
        self.ai_samples_spin = QSpinBox()
        self.ai_samples_spin.setRange(10, 500)
        self.ai_samples_spin.setValue(100)  # Increased default
        settings_layout.addRow("AI samples to generate:", self.ai_samples_spin)
        
        self.ollama_model = QComboBox()
        self.ollama_model.addItems(["gpt-oss:20b", "gpt-oss:120b", "qwen3:8b", "gemma3:12b"])
        settings_layout.addRow("Ollama model:", self.ollama_model)
        
        layout.addWidget(settings_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.collect_human_btn = QPushButton("Collect Human Data")
        self.collect_human_btn.clicked.connect(lambda: self._start_collection(human=True, ai=False))
        self.collect_ai_btn = QPushButton("Generate AI Data")
        self.collect_ai_btn.clicked.connect(lambda: self._start_collection(human=False, ai=True))
        self.collect_all_btn = QPushButton("Collect All")
        self.collect_all_btn.clicked.connect(lambda: self._start_collection(human=True, ai=True))
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_collection)
        
        btn_layout.addWidget(self.collect_human_btn)
        btn_layout.addWidget(self.collect_ai_btn)
        btn_layout.addWidget(self.collect_all_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        
        # PDF Import section
        pdf_group = QGroupBox("PDF Import")
        pdf_layout = QHBoxLayout(pdf_group)
        self.import_pdf_btn = QPushButton("Import PDF Files...")
        self.import_pdf_btn.clicked.connect(self._import_pdfs)
        self.import_pdf_dir_btn = QPushButton("Import PDF Folder...")
        self.import_pdf_dir_btn.clicked.connect(self._import_pdf_directory)
        pdf_layout.addWidget(self.import_pdf_btn)
        pdf_layout.addWidget(self.import_pdf_dir_btn)
        layout.addWidget(pdf_group)
        
        layout.addStretch()
        self.tabs.addTab(tab, "Data Collection")
    
    def _create_training_tab(self):
        """Create the training tab with analytics visualization."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Top row: Settings and Results
        top_row = QHBoxLayout()
        
        # Settings
        settings_group = QGroupBox("Training Settings")
        settings_layout = QFormLayout(settings_group)
        
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["neural_network", "ensemble", "random_forest", "logistic_regression"])
        settings_layout.addRow("Model type:", self.model_type_combo)
        
        # Hyperparameters
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        settings_layout.addRow("Epochs:", self.epochs_spin)
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(0.001)
        settings_layout.addRow("Learning Rate:", self.lr_spin)
        
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(32)
        settings_layout.addRow("Batch Size:", self.batch_spin)
        
        # Initialize values from config
        self._load_training_params()
        
        top_row.addWidget(settings_group)
        
        # Results display
        results_group = QGroupBox("Training Results")
        results_layout = QFormLayout(results_group)
        
        self.accuracy_label = QLabel("-")
        self.accuracy_label.setFont(QFont("Menlo", 14, QFont.Weight.Bold))
        results_layout.addRow("Accuracy:", self.accuracy_label)
        
        self.auc_label = QLabel("-")
        self.auc_label.setFont(QFont("Menlo", 14, QFont.Weight.Bold))
        results_layout.addRow("AUC Score:", self.auc_label)
        
        self.samples_label = QLabel("-")
        results_layout.addRow("Training samples:", self.samples_label)
        
        top_row.addWidget(results_group)
        layout.addLayout(top_row)
        
        # Visualization area (placeholder - will show plots after training)
        viz_group = QGroupBox("Training Analytics")
        viz_layout = QVBoxLayout(viz_group)
        
        self.analytics_display = QTextBrowser()
        self.analytics_display.setFont(QFont("Menlo", 10))
        self.analytics_display.setMinimumHeight(200)
        self.analytics_display.setHtml("<p style='color: #888;'>Train a model to see analytics...</p>")
        viz_layout.addWidget(self.analytics_display)
        
        layout.addWidget(viz_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.train_btn = QPushButton("Train Model")
        self.train_btn.clicked.connect(self._start_training)
        btn_layout.addWidget(self.train_btn)
        layout.addLayout(btn_layout)
        
        self.tabs.addTab(tab, "Training")
    
    def _load_training_params(self):
        """Load training hyperparameters from config."""
        try:
            config_path = os.path.join(PROJECT_ROOT, "config.yaml")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f) or {}
                    nn = cfg.get('training', {}).get('neural_network', {})
                    if 'epochs' in nn:
                        self.epochs_spin.setValue(int(nn['epochs']))
                    if 'learning_rate' in nn:
                        self.lr_spin.setValue(float(nn['learning_rate']))
                    if 'batch_size' in nn:
                        self.batch_spin.setValue(int(nn['batch_size']))
        except Exception as e:
            self._log(f"Error loading training params: {e}")

    def _update_config_params(self):
        """Update config.yaml with current UI values."""
        try:
            config_path = os.path.join(PROJECT_ROOT, "config.yaml")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f) or {}
                
                # Update values
                if 'training' not in cfg: cfg['training'] = {}
                if 'neural_network' not in cfg['training']: cfg['training']['neural_network'] = {}
                
                nn = cfg['training']['neural_network']
                nn['epochs'] = self.epochs_spin.value()
                nn['learning_rate'] = self.lr_spin.value()
                nn['batch_size'] = self.batch_spin.value()
                
                # Write back
                with open(config_path, 'w') as f:
                    yaml.dump(cfg, f, default_flow_style=False)
                
                self._log("Updated config with training parameters")
                
                # Update config editor if it's visible
                if hasattr(self, 'config_editor'):
                     with open(config_path, 'r') as f:
                        self.config_editor.setText(f.read())
        except Exception as e:
            self._log(f"Error updating config: {e}")
    
    def _create_detection_tab(self):
        """Create the AI detection tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Input section
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout(input_group)
        
        # Buttons row
        btn_row = QHBoxLayout()
        self.detect_pdf_btn = QPushButton("Load PDF...")
        self.detect_pdf_btn.clicked.connect(self._load_pdf_for_detection)
        self.detect_text_btn = QPushButton("Analyze Text")
        self.detect_text_btn.clicked.connect(self._analyze_text)
        btn_row.addWidget(self.detect_pdf_btn)
        btn_row.addWidget(self.detect_text_btn)
        btn_row.addStretch()
        input_layout.addLayout(btn_row)
        
        # Text input
        self.detection_input = QTextEdit()
        self.detection_input.setPlaceholderText("Paste text here or load a PDF...")
        self.detection_input.setMaximumHeight(150)
        input_layout.addWidget(self.detection_input)
        
        layout.addWidget(input_group)
        
        # Results section
        results_group = QGroupBox("Detection Results")
        results_layout = QVBoxLayout(results_group)
        
        # Overall score
        score_layout = QHBoxLayout()
        score_layout.addWidget(QLabel("Overall AI Probability:"))
        self.overall_score_label = QLabel("-")
        self.overall_score_label.setFont(QFont("Menlo", 16, QFont.Weight.Bold))
        score_layout.addWidget(self.overall_score_label)
        score_layout.addStretch()
        results_layout.addLayout(score_layout)
        
        # Legend
        legend = QLabel("Legend: <span style='background-color:#ff6b6b;'>High AI (>70%)</span> | "
                       "<span style='background-color:#ffd93d;'>Medium (50-70%)</span> | "
                       "<span style='background-color:#6bcf63;'>Low (<50%)</span>")
        legend.setTextFormat(Qt.TextFormat.RichText)
        results_layout.addWidget(legend)
        
        # Highlighted text display
        self.detection_output = QTextBrowser()
        self.detection_output.setFont(QFont("Georgia", 12))
        self.detection_output.setOpenExternalLinks(False)
        results_layout.addWidget(self.detection_output)
        
        layout.addWidget(results_group)
        
        self.tabs.addTab(tab, "Detection")
    
    def _load_pdf_for_detection(self):
        """Load a PDF file for AI detection."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select PDF", "", "PDF Files (*.pdf)"
        )
        
        if not file_path:
            return
        
        try:
            from pdf_importer import PDFImporter
            importer = PDFImporter()
            text = importer.extract_text(file_path)
            
            if text:
                self.detection_input.setText(text)
                self._log(f"Loaded PDF: {os.path.basename(file_path)}")
            else:
                self._log("No text extracted from PDF")
        except Exception as e:
            self._log(f"Error loading PDF: {e}")
    
    def _analyze_text(self):
        """Analyze text for AI detection with segment highlighting."""
        text = self.detection_input.toPlainText().strip()
        
        if not text:
            QMessageBox.warning(self, "No Text", "Please enter text or load a PDF first.")
            return
        
        self._log("Analyzing text for AI content...")
        
        self._log("Analyzing text for AI content...")
        
        try:
            import sys
            sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
            
            # Try loading Neural Network model first
            from app.services.nn_text_detector import NeuralNetworkDetector
            nn_detector = NeuralNetworkDetector()
            
            # Check if NN model exists
            if os.path.exists(nn_detector.model_path):
                self._log("Using Neural Network model for detection")
                detector = nn_detector
                detector.load_model()
            else:
                # Fallbck to standard ML detector
                from app.services.ml_text_detector import MLTextDetector
                detector = MLTextDetector()
                if not detector.is_trained:
                     QMessageBox.warning(self, "No Model", 
                        "No trained model found. Please train a model first.")
                     return
                self._log("Using Standard ML model (Random Forest/Ensemble)")
            
            # Split text into segments (~50 words each)
            words = text.split()
            segment_size = 50
            segments = []
            
            for i in range(0, len(words), segment_size):
                segment_words = words[i:i + segment_size]
                segments.append(' '.join(segment_words))
            
            # Analyze each segment
            results = []
            total_ai_prob = 0
            
            for segment in segments:
                if len(segment.split()) < 10:  # Skip very short segments
                    results.append({'text': segment, 'ai_prob': 0, 'skip': True})
                    continue
                
                prediction = detector.predict(segment)
                # Handle different key names between detectors
                ai_prob = prediction.get('probability_ai', prediction.get('probability', 0))
                total_ai_prob += ai_prob
                results.append({'text': segment, 'ai_prob': ai_prob, 'skip': False})
            
            # Calculate overall score
            analyzed_count = sum(1 for r in results if not r.get('skip', False))
            overall_prob = total_ai_prob / analyzed_count if analyzed_count > 0 else 0
            
            # Update overall score display
            self.overall_score_label.setText(f"{overall_prob:.1%}")
            if overall_prob > 0.7:
                self.overall_score_label.setStyleSheet("color: #d63031;")
            elif overall_prob > 0.5:
                self.overall_score_label.setStyleSheet("color: #f39c12;")
            else:
                self.overall_score_label.setStyleSheet("color: #27ae60;")
            
            # Build highlighted HTML
            html_parts = []
            for result in results:
                text_segment = result['text']
                ai_prob = result['ai_prob']
                
                if result.get('skip', False):
                    html_parts.append(f"<span>{text_segment}</span> ")
                elif ai_prob > 0.7:
                    html_parts.append(
                        f"<span style='background-color:#ff6b6b;' title='AI: {ai_prob:.1%}'>{text_segment}</span> "
                    )
                elif ai_prob > 0.5:
                    html_parts.append(
                        f"<span style='background-color:#ffd93d;' title='AI: {ai_prob:.1%}'>{text_segment}</span> "
                    )
                else:
                    html_parts.append(
                        f"<span style='background-color:#6bcf63;' title='AI: {ai_prob:.1%}'>{text_segment}</span> "
                    )
            
            html_output = "<p style='line-height:1.8;'>" + "".join(html_parts) + "</p>"
            self.detection_output.setHtml(html_output)
            
            self._log(f"Analysis complete: {len(segments)} segments, overall AI probability: {overall_prob:.1%}")
            
        except Exception as e:
            self._log(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_stats_tab(self):
        """Create the statistics tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Data stats
        data_group = QGroupBox("Training Data Statistics")
        data_layout = QFormLayout(data_group)
        
        self.human_count_label = QLabel("-")
        data_layout.addRow("Human samples:", self.human_count_label)
        
        self.ai_count_label = QLabel("-")
        data_layout.addRow("AI samples:", self.ai_count_label)
        
        self.total_words_label = QLabel("-")
        data_layout.addRow("Total words:", self.total_words_label)
        
        layout.addWidget(data_group)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh Statistics")
        refresh_btn.clicked.connect(self._refresh_stats)
        layout.addWidget(refresh_btn)
        
        layout.addStretch()
        self.tabs.addTab(tab, "Statistics")
    
    def _load_config(self):
        """Load config.yaml into the editor."""
        config_path = os.path.join(PROJECT_ROOT, "config.yaml")
        try:
            with open(config_path, 'r') as f:
                self.config_editor.setText(f.read())
            self._log("Config loaded from config.yaml")
        except Exception as e:
            self._log(f"Error loading config: {e}")
    
    def _save_config(self):
        """Save the editor content to config.yaml."""
        config_path = os.path.join(PROJECT_ROOT, "config.yaml")
        try:
            # Validate YAML
            yaml.safe_load(self.config_editor.toPlainText())
            
            with open(config_path, 'w') as f:
                f.write(self.config_editor.toPlainText())
            self._log("Config saved to config.yaml")
        except yaml.YAMLError as e:
            QMessageBox.warning(self, "Invalid YAML", f"YAML syntax error:\n{e}")
        except Exception as e:
            self._log(f"Error saving config: {e}")
    
    def _start_collection(self, human=True, ai=True):
        """Start data collection in a worker thread."""
        self.collection_worker = DataCollectionWorker(
            collect_human=human,
            collect_ai=ai,
            ai_model=self.ollama_model.currentText(),
            ai_samples=self.ai_samples_spin.value()
        )
        self.collection_worker.progress.connect(self._on_progress)
        self.collection_worker.log.connect(self._log)
        self.collection_worker.finished.connect(self._on_collection_finished)
        
        self._set_collection_buttons_enabled(False)
        self.cancel_btn.setEnabled(True)
        self.collection_worker.start()
    
    def _cancel_collection(self):
        """Cancel the current collection."""
        if self.collection_worker:
            self.collection_worker.cancel()
            self._log("Cancelling collection...")
    
    def _stop_all_workers(self):
        """Stop all running workers (collection and training)."""
        stopped = False
        
        if self.collection_worker and self.collection_worker.isRunning():
            self.collection_worker.cancel()
            self._log("Stopping data collection...")
            stopped = True
        
        if self.training_worker and self.training_worker.isRunning():
            self.training_worker.cancel()
            self._log("Stopping training...")
            stopped = True
        
        if stopped:
            self.status_label.setText("Stopped by user")
            self.progress_bar.setValue(0)
        else:
            self._log("No active workers to stop.")
    
    def _on_collection_finished(self, success, message):
        """Handle collection completion."""
        self._set_collection_buttons_enabled(True)
        self.cancel_btn.setEnabled(False)
        self._log(message)
        self._refresh_stats()
    
    def _set_collection_buttons_enabled(self, enabled):
        """Enable/disable collection buttons."""
        self.collect_human_btn.setEnabled(enabled)
        self.collect_ai_btn.setEnabled(enabled)
        self.collect_all_btn.setEnabled(enabled)
    
    def _start_training(self):
        """Start model training in a worker thread."""
        # Update config with current values
        self._update_config_params()
        
        self.training_worker = TrainingWorker(
            model_type=self.model_type_combo.currentText(),
            epochs=self.epochs_spin.value(),
            learning_rate=self.lr_spin.value(),
            batch_size=self.batch_spin.value()
        )
        self.training_worker.progress.connect(self._on_progress)
        self.training_worker.log.connect(self._log)
        self.training_worker.finished.connect(self._on_training_finished)
        
        self.train_btn.setEnabled(False)
        self.training_worker.start()
    
    def _on_training_finished(self, success, message, results):
        """Handle training completion with detailed analytics."""
        self.train_btn.setEnabled(True)
        self._log(message)
        
        if success and results:
            accuracy = results.get('accuracy', 0)
            auc_score = results.get('auc_score', 0)
            
            self.accuracy_label.setText(f"{accuracy:.1%}")
            self.auc_label.setText(f"{auc_score:.3f}")
            self.samples_label.setText(
                f"{results.get('human_samples', 0)} human + {results.get('ai_samples', 0)} AI"
            )
            
            # Build analytics HTML with confusion matrix
            cm = results.get('confusion_matrix', [[0, 0], [0, 0]])
            training_history = results.get('training_history', {})
            
            # Training history section for neural networks
            history_html = ""
            if training_history and training_history.get('train_loss'):
                epochs = len(training_history['train_loss'])
                final_train = training_history['train_loss'][-1]
                final_val = training_history['val_loss'][-1]
                history_html = f"""
                <h3>Training History ({epochs} epochs)</h3>
                <pre style="font-family: Menlo; background: #1e1e1e; padding: 10px; border-radius: 5px;">
Final Train Loss: {final_train:.4f}
Final Val Loss:   {final_val:.4f}
Train Accuracy:   {training_history['train_acc'][-1]:.3f}
Val Accuracy:     {training_history['val_acc'][-1]:.3f}
                </pre>
                """
            
            html = f"""
            <style>
                pre {{ font-family: Menlo, monospace; margin: 5px 0; }}
                h3 {{ margin: 15px 0 8px 0; color: #ddd; }}
                .metrics {{ background: #2a2a2a; padding: 10px; border-radius: 5px; }}
            </style>
            <h3>Confusion Matrix</h3>
            <pre style="font-family: Menlo; background: #1e1e1e; padding: 10px; border-radius: 5px;">
                 Predicted
              Human    AI
Actual  Human  {cm[0][0]:4d}   {cm[0][1]:4d}
        AI     {cm[1][0]:4d}   {cm[1][1]:4d}

TN={cm[0][0]} (Human→Human)  FP={cm[0][1]} (Human→AI)
FN={cm[1][0]} (AI→Human)     TP={cm[1][1]} (AI→AI)
            </pre>
            {history_html}
            <h3>Performance Summary</h3>
            <pre style="font-family: Menlo; background: #1e1e1e; padding: 10px; border-radius: 5px;">
Accuracy:  {accuracy:.1%}
AUC Score: {auc_score:.3f}
            </pre>
            """
            
            self.analytics_display.setHtml(html)
    
    def _on_progress(self, percent, message):
        """Update progress bar."""
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)
    
    def _log(self, message):
        """Append message to terminal."""
        self.terminal.appendPlainText(f"> {message}")
    
    def _refresh_stats(self):
        """Refresh statistics display."""
        try:
            human_dir = os.path.join(PROJECT_ROOT, "training_data", "human")
            ai_dir = os.path.join(PROJECT_ROOT, "training_data", "ai")
            
            human_count = len([f for f in os.listdir(human_dir) if f.endswith('.txt')]) if os.path.exists(human_dir) else 0
            ai_count = len([f for f in os.listdir(ai_dir) if f.endswith('.txt')]) if os.path.exists(ai_dir) else 0
            
            self.human_count_label.setText(str(human_count))
            self.ai_count_label.setText(str(ai_count))
            self.total_words_label.setText("-")  # TODO: Calculate total words
            
        except Exception as e:
            self._log(f"Error refreshing stats: {e}")
    
    def _import_pdfs(self):
        """Import selected PDF files."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select PDF Files",
            "",
            "PDF Files (*.pdf)"
        )
        
        if not files:
            return
        
        self._log(f"Importing {len(files)} PDF files...")
        
        try:
            from pdf_importer import PDFImporter
            importer = PDFImporter()
            
            imported = 0
            for pdf_path in files:
                result = importer.import_pdf(pdf_path)
                if result:
                    self._log(f"  Imported: {result['word_count']} words from {os.path.basename(pdf_path)}")
                    imported += 1
            
            self._log(f"Successfully imported {imported}/{len(files)} PDF files")
            self._refresh_stats()
            
        except Exception as e:
            self._log(f"Error importing PDFs: {e}")
    
    def _import_pdf_directory(self):
        """Import all PDFs from a directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select PDF Directory"
        )
        
        if not directory:
            return
        
        self._log(f"Importing PDFs from: {directory}")
        
        try:
            from pdf_importer import PDFImporter
            importer = PDFImporter()
            results = importer.import_directory(directory)
            
            self._log(f"Successfully imported {len(results)} PDF files")
            self._refresh_stats()
            
        except Exception as e:
            self._log(f"Error importing PDF directory: {e}")
