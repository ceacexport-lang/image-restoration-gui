import sys
import threading
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QTabWidget,
)

from ai_pipeline import enhance_image


class ProcessSignals(QObject):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    done = pyqtSignal(bool, str)


class OldPhotoRestorationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Old Photo Restoration Studio v2")
        self.resize(1200, 800)

        self.input_path = ""
        self.input_folder = ""
        self.output_path = ""
        self.signals = ProcessSignals()
        self.signals.log.connect(self._on_log)
        self.signals.progress.connect(self._on_progress)
        self.signals.done.connect(self._on_done)

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)

        # Tabs for single image and batch processing
        self.tabs = QTabWidget()

        # Tab 1: Single image with presets
        self.tab_single = self._build_single_image_tab()
        self.tabs.addTab(self.tab_single, "Single Image")

        # Tab 2: Batch processing
        self.tab_batch = self._build_batch_tab()
        self.tabs.addTab(self.tab_batch, "Batch Restore")

        layout.addWidget(self.tabs)
        self.setCentralWidget(central)

    def _build_single_image_tab(self):
        widget = QWidget()
        main_layout = QVBoxLayout()

        top_row = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()

        # Input section
        input_group = QGroupBox("Input")
        input_layout = QFormLayout()
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setReadOnly(True)
        self.image_path_edit.setPlaceholderText("No image selected")
        input_layout.addRow("Image:", self.image_path_edit)

        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        input_layout.addRow(load_btn)
        input_group.setLayout(input_layout)
        left.addWidget(input_group)

        # Remote API section
        remote_group = QGroupBox("Remote Processing (Optional)")
        remote_layout = QFormLayout()
        self.remote_api_edit = QLineEdit()
        self.remote_api_edit.setPlaceholderText("e.g., https://api.replicate.com/...")
        remote_layout.addRow("API URL:", self.remote_api_edit)
        self.use_remote_box = QCheckBox("Use remote API")
        remote_layout.addRow(self.use_remote_box)
        remote_group.setLayout(remote_layout)
        left.addWidget(remote_group)

        # Quick presets
        preset_group = QGroupBox("Quick Restore Presets")
        preset_layout = QVBoxLayout()
        self.restore_btn = QPushButton("🎯 Restore Old Photo (Auto)")
        self.restore_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        self.restore_btn.clicked.connect(self.restore_old_photo_quick)
        preset_layout.addWidget(self.restore_btn)
        preset_group.setLayout(preset_layout)
        left.addWidget(preset_group)

        # Advanced controls
        adv_group = QGroupBox("Advanced Controls")
        adv_layout = QFormLayout()

        # Scale
        self.scale_box = QComboBox()
        self.scale_box.addItems(["2x", "4x"])
        self.scale_box.setCurrentIndex(0)
        adv_layout.addRow("Upscale:", self.scale_box)

        # Workflow
        self.workflow_box = QComboBox()
        self.workflow_box.addItems([
            "Balanced Restore",
            "Faded Color Restore",
            "Modern Fresh Look",
            "Portrait Refresh",
            "High Detail Restore",
        ])
        adv_layout.addRow("Workflow:", self.workflow_box)

        # Denoise
        self.denoise_box = QCheckBox("Denoise")
        self.denoise_box.setChecked(True)
        adv_layout.addRow(self.denoise_box)

        # Color preset slider (0-10 scale)
        color_label = QLabel("Color Enhancement")
        self.color_slider = QSlider(Qt.Horizontal)
        self.color_slider.setRange(50, 250)  # 0.5x to 2.5x
        self.color_slider.setValue(120)  # 1.2x default
        self.color_slider.setTickInterval(10)
        self.color_slider.setTickPosition(QSlider.TicksBelow)
        self.color_value_label = QLabel("1.20x")
        self.color_slider.sliderMoved.connect(lambda: self.color_value_label.setText(f"{self.color_slider.value() / 100:.2f}x"))
        color_row = QHBoxLayout()
        color_row.addWidget(self.color_slider)
        color_row.addWidget(self.color_value_label)
        adv_layout.addRow(color_label, color_row)

        # Face preset slider
        face_label = QLabel("Face Enhancement")
        self.face_slider = QSlider(Qt.Horizontal)
        self.face_slider.setRange(0, 100)  # 0 = no, 100 = full
        self.face_slider.setValue(0)
        self.face_slider.setTickInterval(10)
        self.face_slider.setTickPosition(QSlider.TicksBelow)
        self.face_value_label = QLabel("Off")
        self.face_slider.sliderMoved.connect(lambda: self._update_face_label())
        face_row = QHBoxLayout()
        face_row.addWidget(self.face_slider)
        face_row.addWidget(self.face_value_label)
        adv_layout.addRow(face_label, face_row)

        # Detail preset slider
        detail_label = QLabel("Detail Enhancement")
        self.detail_slider = QSlider(Qt.Horizontal)
        self.detail_slider.setRange(50, 250)  # 0.5x to 2.5x
        self.detail_slider.setValue(120)  # 1.2x default
        self.detail_slider.setTickInterval(10)
        self.detail_slider.setTickPosition(QSlider.TicksBelow)
        self.detail_value_label = QLabel("1.20x")
        self.detail_slider.sliderMoved.connect(lambda: self.detail_value_label.setText(f"{self.detail_slider.value() / 100:.2f}x"))
        detail_row = QHBoxLayout()
        detail_row.addWidget(self.detail_slider)
        detail_row.addWidget(self.detail_value_label)
        adv_layout.addRow(detail_label, detail_row)

        adv_group.setLayout(adv_layout)
        left.addWidget(adv_group)

        # Action buttons
        action_row = QHBoxLayout()
        run_btn = QPushButton("Run Enhancement")
        run_btn.clicked.connect(self.run_pipeline)
        save_btn = QPushButton("Save Output")
        save_btn.clicked.connect(self.save_output)
        action_row.addWidget(run_btn)
        action_row.addWidget(save_btn)
        left.addLayout(action_row)

        # Log section
        log_group = QGroupBox("Process Log")
        log_layout = QVBoxLayout()
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(120)
        log_layout.addWidget(self.log_edit)
        log_group.setLayout(log_layout)
        left.addWidget(log_group)

        # Preview section
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()

        before_label = QLabel("Before")
        before_label.setAlignment(Qt.AlignCenter)
        before_label.setMinimumHeight(320)
        before_label.setStyleSheet("border: 1px solid #888; background: #202020;")

        after_label = QLabel("After")
        after_label.setAlignment(Qt.AlignCenter)
        after_label.setMinimumHeight(320)
        after_label.setStyleSheet("border: 1px solid #888; background: #202020;")

        self.before_label = before_label
        self.after_label = after_label

        preview_row = QHBoxLayout()
        preview_row.addWidget(before_label)
        preview_row.addWidget(after_label)
        preview_layout.addLayout(preview_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        preview_layout.addWidget(self.progress_bar)
        preview_group.setLayout(preview_layout)
        right.addWidget(preview_group)

        top_row.addLayout(left, 1)
        top_row.addLayout(right, 1)
        main_layout.addLayout(top_row)
        widget.setLayout(main_layout)
        return widget

    def _build_batch_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()

        folder_group = QGroupBox("Batch Input")
        folder_layout = QFormLayout()
        self.batch_folder_edit = QLineEdit()
        self.batch_folder_edit.setReadOnly(True)
        self.batch_folder_edit.setPlaceholderText("No folder selected")
        folder_layout.addRow("Folder:", self.batch_folder_edit)
        select_btn = QPushButton("Select Folder")
        select_btn.clicked.connect(self.select_batch_folder)
        folder_layout.addRow(select_btn)
        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)

        output_group = QGroupBox("Batch Output")
        output_layout = QFormLayout()
        self.batch_output_edit = QLineEdit()
        self.batch_output_edit.setPlaceholderText("Leave blank for 'restored' subfolder")
        output_layout.addRow("Output Folder:", self.batch_output_edit)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        options_group = QGroupBox("Batch Options")
        options_layout = QFormLayout()
        self.batch_scale_box = QComboBox()
        self.batch_scale_box.addItems(["2x", "4x"])
        options_layout.addRow("Upscale:", self.batch_scale_box)
        self.batch_denoise_box = QCheckBox("Denoise all")
        self.batch_denoise_box.setChecked(True)
        options_layout.addRow(self.batch_denoise_box)
        self.batch_workflow_box = QComboBox()
        self.batch_workflow_box.addItems(["Balanced Restore", "Faded Color Restore", "Modern Fresh Look"])
        options_layout.addRow("Workflow:", self.batch_workflow_box)
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        action_group = QGroupBox("Actions")
        action_layout = QHBoxLayout()
        start_batch_btn = QPushButton("Start Batch Restore")
        start_batch_btn.clicked.connect(self.start_batch_restore)
        action_layout.addWidget(start_batch_btn)
        action_group.setLayout(action_layout)
        layout.addWidget(action_group)

        self.batch_log = QTextEdit()
        self.batch_log.setReadOnly(True)
        layout.addWidget(self.batch_log)

        self.batch_progress = QProgressBar()
        self.batch_progress.setValue(0)
        layout.addWidget(self.batch_progress)

        widget.setLayout(layout)
        return widget

    def _update_face_label(self):
        val = self.face_slider.value()
        if val == 0:
            self.face_value_label.setText("Off")
        elif val < 50:
            self.face_value_label.setText("Light")
        else:
            self.face_value_label.setText("Strong")

    def log(self, message: str):
        self.log_edit.append(message)
        QApplication.processEvents()

    def _on_log(self, message: str):
        self.log(message)

    def _on_progress(self, value: int):
        self.progress_bar.setValue(value)

    def _on_done(self, success: bool, output_path: str):
        if success and Path(output_path).exists():
            pixmap = QPixmap(output_path)
            if not pixmap.isNull():
                self.after_label.setPixmap(pixmap.scaledToWidth(420, Qt.SmoothTransformation))
        QMessageBox.information(self, "Done", "Image restoration completed.")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return

        self.input_path = path
        self.output_path = str(Path(path).with_name(Path(path).stem + "_restored.png"))
        self.image_path_edit.setText(path)

        pixmap = QPixmap(path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Load failed", "Could not load the selected file.")
            return

        self.before_label.setPixmap(pixmap.scaledToWidth(420, Qt.SmoothTransformation))
        self.log(f"Loaded image: {path}")

    def restore_old_photo_quick(self):
        """One-click auto restore with sensible defaults for old photos."""
        if not self.input_path:
            QMessageBox.warning(self, "No image", "Please load an image first.")
            return

        self.scale_box.setCurrentText("4x")
        self.workflow_box.setCurrentText("Faded Color Restore")
        self.denoise_box.setChecked(True)
        self.color_slider.setValue(135)  # 1.35x for faded colors
        self.detail_slider.setValue(130)  # 1.30x sharpness
        self.face_slider.setValue(50)  # Light face restoration

        self.log("Applied quick restore preset: 4x upscale, faded color mode, light face restoration.")
        self.run_pipeline()

    def run_pipeline(self):
        if not self.input_path:
            QMessageBox.warning(self, "No image", "Please load an image first.")
            return

        self.progress_bar.setValue(0)
        self.log("Starting image enhancement...")

        def process():
            try:
                scale = 4 if self.scale_box.currentText() == "4x" else 2
                workflow = self.workflow_box.currentText()
                denoise = self.denoise_box.isChecked()
                color_boost = self.color_slider.value() / 100.0
                sharpness = self.detail_slider.value() / 100.0
                face_restore = self.face_slider.value() > 0
                use_remote = self.use_remote_box.isChecked()
                remote_api_url = self.remote_api_edit.text().strip() if use_remote else ""

                self.signals.progress.emit(20)
                ok, output, message = enhance_image(
                    self.input_path,
                    self.output_path,
                    scale=scale,
                    denoise=denoise,
                    color_boost=color_boost,
                    contrast=1.10,
                    sharpness=sharpness,
                    face_restore=face_restore,
                    workflow=workflow,
                    use_remote=use_remote,
                    remote_api_url=remote_api_url,
                )
                self.signals.progress.emit(100)
                self.signals.log.emit(f"{message}")
                self.signals.log.emit(f"Output saved to: {output}")
                self.signals.done.emit(ok, output)
            except Exception as exc:
                self.signals.log.emit(f"Error: {exc}")
                QMessageBox.critical(self, "Processing Error", str(exc))

        thread = threading.Thread(target=process, daemon=True)
        thread.start()

    def save_output(self):
        if not self.output_path or not Path(self.output_path).exists():
            QMessageBox.warning(self, "No output", "Run enhancement first to generate an output image.")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save restored image",
            str(Path(self.output_path).with_suffix(".png")),
            "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tif)",
        )
        if not save_path:
            return

        from PIL import Image
        img = Image.open(self.output_path)
        img.save(save_path)
        self.log(f"Saved final result: {save_path}")
        QMessageBox.information(self, "Saved", "Final image saved successfully.")

    def select_batch_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with images")
        if folder:
            self.input_folder = folder
            self.batch_folder_edit.setText(folder)

    def start_batch_restore(self):
        if not self.input_folder:
            QMessageBox.warning(self, "No folder", "Please select a folder with images.")
            return

        def batch_process():
            try:
                folder = Path(self.input_folder)
                output_folder = Path(self.batch_output_edit.text()) if self.batch_output_edit.text() else folder / "restored"
                output_folder.mkdir(parents=True, exist_ok=True)

                image_files = list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
                total = len(image_files)

                self.batch_log.append(f"Found {total} images to process.")
                QApplication.processEvents()

                scale = 4 if self.batch_scale_box.currentText() == "4x" else 2
                workflow = self.batch_workflow_box.currentText()
                denoise = self.batch_denoise_box.isChecked()

                for idx, img_file in enumerate(image_files):
                    output_file = output_folder / f"{img_file.stem}_restored.png"
                    self.batch_log.append(f"Processing {idx + 1}/{total}: {img_file.name}...")
                    QApplication.processEvents()

                    ok, out, msg = enhance_image(
                        str(img_file),
                        str(output_file),
                        scale=scale,
                        denoise=denoise,
                        color_boost=1.2,
                        contrast=1.1,
                        sharpness=1.2,
                        face_restore=False,
                        workflow=workflow,
                    )
                    self.batch_log.append(f"✓ {img_file.name}: {msg}")
                    self.batch_progress.setValue(int((idx + 1) / total * 100))
                    QApplication.processEvents()

                self.batch_log.append(f"\nBatch complete! Output: {output_folder}")
                QMessageBox.information(self, "Done", f"Batch processing complete! Output folder: {output_folder}")
            except Exception as exc:
                self.batch_log.append(f"Error: {exc}")
                QMessageBox.critical(self, "Batch Error", str(exc))

        thread = threading.Thread(target=batch_process, daemon=True)
        thread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OldPhotoRestorationApp()
    window.show()
    sys.exit(app.exec_())
