import sys
import threading
from pathlib import Path

import cv2
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ai_pipeline import enhance_image


class ProcessSignals(QObject):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    done = pyqtSignal(bool, str)


class OldPhotoRestorationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Old Photo Restoration Studio")
        self.resize(1400, 900)

        self.input_path = ""
        self.input_folder = ""
        self.output_path = ""
        self.batch_processed_files = []

        self.signals = ProcessSignals()
        self.signals.log.connect(self._on_log)
        self.signals.progress.connect(self._on_progress)
        self.signals.done.connect(self._on_done)

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        main_layout = QHBoxLayout()

        # Left panel: controls
        left = QVBoxLayout()

        # Input section - Load Image or Folder side by side
        input_group = QGroupBox("Load Image / Folder")
        input_layout = QHBoxLayout()

        # Load Image column
        img_col = QVBoxLayout()
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setReadOnly(True)
        self.image_path_edit.setPlaceholderText("No image selected")
        img_col.addWidget(QLabel("Image:"))
        img_col.addWidget(self.image_path_edit)
        load_img_btn = QPushButton("Load Image")
        load_img_btn.clicked.connect(self.load_image)
        img_col.addWidget(load_img_btn)

        # Load Folder column
        folder_col = QVBoxLayout()
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setReadOnly(True)
        self.folder_path_edit.setPlaceholderText("No folder selected")
        folder_col.addWidget(QLabel("Folder:"))
        folder_col.addWidget(self.folder_path_edit)
        load_folder_btn = QPushButton("Load Folder")
        load_folder_btn.clicked.connect(self.load_folder)
        folder_col.addWidget(load_folder_btn)

        input_layout.addLayout(img_col)
        input_layout.addLayout(folder_col)
        input_group.setLayout(input_layout)
        left.addWidget(input_group)

        # Remote API section
        remote_group = QGroupBox("Remote Processing (Optional)")
        remote_layout = QFormLayout()
        self.remote_api_edit = QLineEdit()
        self.remote_api_edit.setPlaceholderText("e.g. https://api.example.com/restore")
        remote_layout.addRow("API URL:", self.remote_api_edit)
        self.use_remote_box = QCheckBox("Use remote AI")
        remote_layout.addRow(self.use_remote_box)
        remote_group.setLayout(remote_layout)
        left.addWidget(remote_group)

        # One-Click Restore
        preset_group = QGroupBox("One-Click Restore")
        preset_layout = QVBoxLayout()
        self.restore_btn = QPushButton("🎯 Restore Old Photo")
        self.restore_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #2196F3; color: white;")
        self.restore_btn.clicked.connect(self.restore_old_photo_quick)
        preset_layout.addWidget(self.restore_btn)
        preset_group.setLayout(preset_layout)
        left.addWidget(preset_group)

        # Parameters
        adv_group = QGroupBox("Restoration Parameters")
        adv_layout = QFormLayout()

        self.scale_box = QComboBox()
        self.scale_box.addItems(["2x", "4x"])
        self.scale_box.setCurrentIndex(1)
        adv_layout.addRow("Upscale:", self.scale_box)

        self.workflow_box = QComboBox()
        self.workflow_box.addItems([
            "Balanced Restore",
            "Faded Color Restore",
            "Modern Fresh Look",
            "Portrait Refresh",
            "High Detail Restore",
        ])
        self.workflow_box.setCurrentIndex(1)
        adv_layout.addRow("Workflow:", self.workflow_box)

        self.denoise_box = QCheckBox("Denoise")
        self.denoise_box.setChecked(True)
        adv_layout.addRow(self.denoise_box)

        self.color_slider = QSlider(Qt.Horizontal)
        self.color_slider.setRange(50, 250)
        self.color_slider.setValue(135)
        self.color_value_label = QLabel("1.35x")
        self.color_slider.valueChanged.connect(lambda: self.color_value_label.setText(f"{self.color_slider.value()/100:.2f}x"))
        color_row = QHBoxLayout()
        color_row.addWidget(self.color_slider)
        color_row.addWidget(self.color_value_label)
        adv_layout.addRow("Color:", color_row)

        self.face_slider = QSlider(Qt.Horizontal)
        self.face_slider.setRange(0, 100)
        self.face_slider.setValue(50)
        self.face_value_label = QLabel("Light")
        self.face_slider.valueChanged.connect(self._update_face_label)
        face_row = QHBoxLayout()
        face_row.addWidget(self.face_slider)
        face_row.addWidget(self.face_value_label)
        adv_layout.addRow("Face:", face_row)

        self.detail_slider = QSlider(Qt.Horizontal)
        self.detail_slider.setRange(50, 250)
        self.detail_slider.setValue(130)
        self.detail_value_label = QLabel("1.30x")
        self.detail_slider.valueChanged.connect(lambda: self.detail_value_label.setText(f"{self.detail_slider.value()/100:.2f}x"))
        detail_row = QHBoxLayout()
        detail_row.addWidget(self.detail_slider)
        detail_row.addWidget(self.detail_value_label)
        adv_layout.addRow("Detail:", detail_row)

        adv_group.setLayout(adv_layout)
        left.addWidget(adv_group)

        # Action buttons
        action_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Enhancement")
        self.run_btn.clicked.connect(self.run_pipeline)
        save_btn = QPushButton("Save Output")
        save_btn.clicked.connect(self.save_output)
        action_row.addWidget(self.run_btn)
        action_row.addWidget(save_btn)
        left.addLayout(action_row)

        # Create Video button
        self.create_video_btn = QPushButton("🎬 Create & Play Video")
        self.create_video_btn.setStyleSheet("font-weight: bold; padding: 8px; background-color: #4CAF50; color: white;")
        self.create_video_btn.setEnabled(False)
        self.create_video_btn.clicked.connect(self.create_and_play_video)
        left.addWidget(self.create_video_btn)

        # Log section
        log_group = QGroupBox("Process Log")
        log_layout = QVBoxLayout()
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(150)
        log_layout.addWidget(self.log_edit)
        log_group.setLayout(log_layout)
        left.addWidget(log_group)

        left.addStretch()

        # Right panel: preview
        right = QVBoxLayout()
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()

        before_label = QLabel("Before")
        before_label.setAlignment(Qt.AlignCenter)
        before_label.setMinimumHeight(350)
        before_label.setStyleSheet("border: 1px solid #888; background: #202020; color: #888;")

        after_label = QLabel("After")
        after_label.setAlignment(Qt.AlignCenter)
        after_label.setMinimumHeight(350)
        after_label.setStyleSheet("border: 1px solid #888; background: #202020; color: #888;")

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

        main_layout.addLayout(left, 1)
        main_layout.addLayout(right, 1.5)
        layout.addLayout(main_layout)
        self.setCentralWidget(central)

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
                self.after_label.setPixmap(pixmap.scaledToWidth(500, Qt.SmoothTransformation))

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path:
            return

        self.input_path = path
        self.input_folder = ""
        self.image_path_edit.setText(path)
        self.folder_path_edit.setText("")
        self.output_path = str(Path(path).with_name(f"{Path(path).stem}_restored.png"))
        self.batch_processed_files = []
        self.create_video_btn.setEnabled(False)

        pixmap = QPixmap(path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Load failed", "Could not load the selected image.")
            return
        self.before_label.setPixmap(pixmap.scaledToWidth(500, Qt.SmoothTransformation))
        self.after_label.setText("After")
        self.log(f"Loaded image: {path}")

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with images")
        if not folder:
            return

        self.input_folder = folder
        self.input_path = ""
        self.folder_path_edit.setText(folder)
        self.image_path_edit.setText("")
        self.create_video_btn.setEnabled(False)
        self.batch_processed_files = []
        self.before_label.setText("Folder loaded")
        self.after_label.setText("After")

        files = list(Path(folder).glob("*.png")) + list(Path(folder).glob("*.jpg")) + list(Path(folder).glob("*.jpeg"))
        self.log(f"Loaded folder: {folder}")
        self.log(f"Found {len(files)} images")

    def restore_old_photo_quick(self):
        if not self.input_path and not self.input_folder:
            QMessageBox.warning(self, "No input", "Please load an image or folder first.")
            return

        self.scale_box.setCurrentText("4x")
        self.workflow_box.setCurrentText("Faded Color Restore")
        self.denoise_box.setChecked(True)
        self.color_slider.setValue(135)
        self.face_slider.setValue(50)
        self.detail_slider.setValue(130)
        self.log("Applied quick restore preset: 4x, Faded Color, Light Face.")
        self.run_pipeline()

    def run_pipeline(self):
        if not self.input_path and not self.input_folder:
            QMessageBox.warning(self, "No input", "Please load an image or folder first.")
            return

        self.progress_bar.setValue(0)
        self.log("Starting restoration...")

        def worker():
            try:
                scale = 4 if self.scale_box.currentText() == "4x" else 2
                workflow = self.workflow_box.currentText()
                denoise = self.denoise_box.isChecked()
                color_boost = self.color_slider.value() / 100.0
                sharpness = self.detail_slider.value() / 100.0
                face_restore = self.face_slider.value() > 0
                use_remote = self.use_remote_box.isChecked()
                remote_api_url = self.remote_api_edit.text().strip() if use_remote else ""

                if self.input_path:
                    # Single image mode
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
                    self.signals.log.emit(message)
                    self.signals.log.emit(f"Output: {output}")
                    self.signals.done.emit(ok, output)
                else:
                    # Folder mode - process all images
                    self._process_folder_batch(scale, denoise, color_boost, sharpness, face_restore, workflow, use_remote, remote_api_url)
            except Exception as exc:
                self.signals.log.emit(f"Error: {exc}")
                QMessageBox.critical(self, "Processing Error", str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _process_folder_batch(self, scale, denoise, color_boost, sharpness, face_restore, workflow, use_remote, remote_api_url):
        """Process all images in the folder using selected parameters."""
        folder = Path(self.input_folder)
        output_folder = folder / "restored"
        output_folder.mkdir(parents=True, exist_ok=True)

        image_files = sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")))
        total = len(image_files)
        self.batch_processed_files = []

        self.signals.log.emit(f"Processing {total} images from {folder.name}")
        for idx, img_file in enumerate(image_files):
            output_file = output_folder / f"{img_file.stem}_restored.png"
            self.signals.log.emit(f"[{idx + 1}/{total}] {img_file.name}")
            
            ok, out, msg = enhance_image(
                str(img_file),
                str(output_file),
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
            if ok and Path(out).exists():
                self.batch_processed_files.append(str(out))
                self.signals.log.emit(f"  ✓ Done")
            else:
                self.signals.log.emit(f"  ✗ Failed")
            
            self.signals.progress.emit(int((idx + 1) / total * 100))
            QApplication.processEvents()

        self.signals.log.emit(f"\n✓ Processed {len(self.batch_processed_files)}/{total} images")
        self.signals.log.emit(f"Output: {output_folder}")
        self.create_video_btn.setEnabled(len(self.batch_processed_files) > 0)

    def create_and_play_video(self):
        """Create video from processed images and play it."""
        if not self.batch_processed_files:
            QMessageBox.warning(self, "No processed images", "Please process a folder first.")
            return

        output_folder = Path(self.input_folder) / "restored"
        video_path = output_folder / "restoration_sequence.mp4"

        try:
            self.log(f"Creating video from {len(self.batch_processed_files)} images...")
            QApplication.processEvents()

            first = cv2.imread(self.batch_processed_files[0])
            if first is None:
                raise ValueError("Could not read first image")
            
            height, width = first.shape[:2]
            fps = 2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

            for idx, img_path in enumerate(self.batch_processed_files):
                frame = cv2.imread(img_path)
                if frame is not None:
                    writer.write(frame)
                self.log(f"  Frame {idx + 1}/{len(self.batch_processed_files)}")
                QApplication.processEvents()

            writer.release()
            self.log(f"✓ Video created: {video_path}")
            QMessageBox.information(self, "Video Ready", f"Video saved to:\n{video_path}\n\nStarting playback...")

            # Play video
            self._play_video(str(video_path))
        except Exception as exc:
            self.log(f"Video error: {exc}")
            QMessageBox.critical(self, "Video Error", str(exc))

    def _play_video(self, video_path: str):
        """Play video using system player."""
        try:
            import subprocess
            import platform
            
            system = platform.system()
            if system == "Linux":
                subprocess.Popen(["vlc", video_path])
            elif system == "Darwin":
                subprocess.Popen(["open", video_path])
            elif system == "Windows":
                subprocess.Popen(["powershell", "-Command", f"Start-Process '{video_path}'"])
            else:
                self.log("Please open the video file manually.")
        except Exception as exc:
            self.log(f"Playback error: {exc}. Please open manually: {video_path}")

    def save_output(self):
        if not self.output_path or not Path(self.output_path).exists():
            QMessageBox.warning(self, "No output", "Run enhancement first to generate output.")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Save restored image", str(Path(self.output_path).with_suffix(".png")), "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tif)")
        if not save_path:
            return

        from PIL import Image
        image = Image.open(self.output_path)
        image.save(save_path)
        self.log(f"Saved: {save_path}")
        QMessageBox.information(self, "Saved", "Image saved successfully.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OldPhotoRestorationApp()
    window.show()
    sys.exit(app.exec_())
