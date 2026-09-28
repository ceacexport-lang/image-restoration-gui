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
        self.setWindowTitle("Old Photo Restoration Studio v3")
        self.resize(1200, 900)

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

        self.tabs = QTabWidget()
        self.tab_single = self._build_single_tab()
        self.tabs.addTab(self.tab_single, "Single + Folder")
        self.tab_batch = self._build_batch_tab()
        self.tabs.addTab(self.tab_batch, "Batch Restore")

        layout.addWidget(self.tabs)
        self.setCentralWidget(central)

    def _build_single_tab(self):
        widget = QWidget()
        main_layout = QVBoxLayout()

        top_row = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()

        input_group = QGroupBox("Load Image / Folder")
        input_layout = QFormLayout()

        self.image_path_edit = QLineEdit()
        self.image_path_edit.setReadOnly(True)
        self.image_path_edit.setPlaceholderText("No image selected")
        input_layout.addRow("Image:", self.image_path_edit)

        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        input_layout.addRow(load_btn)

        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setReadOnly(True)
        self.folder_path_edit.setPlaceholderText("No folder selected")
        input_layout.addRow("Folder:", self.folder_path_edit)

        load_folder_btn = QPushButton("Load Folder")
        load_folder_btn.clicked.connect(self.load_folder)
        input_layout.addRow(load_folder_btn)

        input_group.setLayout(input_layout)
        left.addWidget(input_group)

        remote_group = QGroupBox("Remote Processing (Optional)")
        remote_layout = QFormLayout()
        self.remote_api_edit = QLineEdit()
        self.remote_api_edit.setPlaceholderText("e.g. https://api.example.com/restore")
        remote_layout.addRow("API URL:", self.remote_api_edit)
        self.use_remote_box = QCheckBox("Use remote AI")
        remote_layout.addRow(self.use_remote_box)
        remote_group.setLayout(remote_layout)
        left.addWidget(remote_group)

        preset_group = QGroupBox("One-Click Restore")
        preset_layout = QVBoxLayout()
        self.restore_btn = QPushButton("🎯 Restore Old Photo")
        self.restore_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        self.restore_btn.clicked.connect(self.restore_old_photo_quick)
        preset_layout.addWidget(self.restore_btn)
        preset_group.setLayout(preset_layout)
        left.addWidget(preset_group)

        adv_group = QGroupBox("Parameters")
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

        action_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Enhancement")
        self.run_btn.clicked.connect(self.run_pipeline)
        save_btn = QPushButton("Save Output")
        save_btn.clicked.connect(self.save_output)
        action_row.addWidget(self.run_btn)
        action_row.addWidget(save_btn)
        left.addLayout(action_row)

        self.create_video_btn = QPushButton("🎬 Create & Play Video")
        self.create_video_btn.setEnabled(False)
        self.create_video_btn.clicked.connect(self.create_and_play_video)
        left.addWidget(self.create_video_btn)

        log_group = QGroupBox("Process Log")
        log_layout = QVBoxLayout()
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(120)
        log_layout.addWidget(self.log_edit)
        log_group.setLayout(log_layout)
        left.addWidget(log_group)

        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()

        before_label = QLabel("Before")
        before_label.setAlignment(Qt.AlignCenter)
        before_label.setMinimumHeight(300)
        before_label.setStyleSheet("border: 1px solid #888; background: #202020;")

        after_label = QLabel("After")
        after_label.setAlignment(Qt.AlignCenter)
        after_label.setMinimumHeight(300)
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

        folder_group = QGroupBox("Batch Folder")
        folder_layout = QFormLayout()
        self.batch_folder_edit = QLineEdit()
        self.batch_folder_edit.setReadOnly(True)
        self.batch_folder_edit.setPlaceholderText("No folder selected")
        folder_layout.addRow("Folder:", self.batch_folder_edit)
        btn = QPushButton("Select Folder")
        btn.clicked.connect(self.select_batch_folder)
        folder_layout.addRow(btn)
        folder_group.setLayout(folder_layout)
        layout.addWidget(folder_group)

        output_group = QGroupBox("Batch Output")
        output_layout = QFormLayout()
        self.batch_output_edit = QLineEdit()
        self.batch_output_edit.setPlaceholderText("Optional output folder")
        output_layout.addRow("Output:", self.batch_output_edit)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        self.batch_scale_box = QComboBox()
        self.batch_scale_box.addItems(["2x", "4x"])
        self.batch_scale_box.setCurrentIndex(1)

        self.batch_denoise_box = QCheckBox("Denoise all")
        self.batch_denoise_box.setChecked(True)

        self.batch_workflow_box = QComboBox()
        self.batch_workflow_box.addItems(["Balanced Restore", "Faded Color Restore", "Modern Fresh Look"])

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
        self.before_label.setPixmap(pixmap.scaledToWidth(420, Qt.SmoothTransformation))
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

        files = list(Path(folder).glob("*.png")) + list(Path(folder).glob("*.jpg")) + list(Path(folder).glob("*.jpeg"))
        self.log(f"Loaded folder: {folder} ({len(files)} images found)")

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
        self.log("Applied quick restore preset.")
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
                    self._process_folder_batch(scale, denoise, color_boost, sharpness, face_restore, workflow, use_remote, remote_api_url)
            except Exception as exc:
                self.signals.log.emit(f"Error: {exc}")
                QMessageBox.critical(self, "Processing Error", str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _process_folder_batch(self, scale, denoise, color_boost, sharpness, face_restore, workflow, use_remote, remote_api_url):
        folder = Path(self.input_folder)
        output_folder = folder / "restored"
        output_folder.mkdir(parents=True, exist_ok=True)

        image_files = sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")))
        total = len(image_files)
        self.batch_processed_files = []

        self.signals.log.emit(f"Found {total} images in {folder}")
        for idx, img_file in enumerate(image_files):
            output_file = output_folder / f"{img_file.stem}_restored.png"
            self.signals.log.emit(f"Processing {idx + 1}/{total}: {img_file.name}")
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
                self.signals.log.emit(f"✓ {img_file.name}: processed")
            else:
                self.signals.log.emit(f"✗ {img_file.name}: failed")
            self.signals.progress.emit(int((idx + 1) / total * 100))
            QApplication.processEvents()

        self.signals.log.emit(f"Folder complete. Output in: {output_folder}")
        self.create_video_btn.setEnabled(len(self.batch_processed_files) > 0)

    def create_and_play_video(self):
        if not self.batch_processed_files:
            QMessageBox.warning(self, "No processed output", "Please process a folder first to create a video.")
            return

        output_folder = Path(self.input_folder) / "restored"
        video_path = output_folder / "restoration_sequence.mp4"

        try:
            first = cv2.imread(self.batch_processed_files[0])
            if first is None:
                raise ValueError("Could not read first image for video creation")
            height, width = first.shape[:2]
            fps = 2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

            for img_path in self.batch_processed_files:
                frame = cv2.imread(img_path)
                if frame is not None:
                    writer.write(frame)

            writer.release()
            self.log(f"Video created: {video_path}")
            QMessageBox.information(self, "Video Ready", f"Saved to:\n{video_path}")

            import subprocess
            import platform
            if platform.system() == "Linux":
                subprocess.Popen(["vlc", str(video_path)])
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(video_path)])
            elif platform.system() == "Windows":
                subprocess.Popen(["powershell", "-Command", f"Start-Process '{video_path}'"])
        except Exception as exc:
            self.log(f"Video error: {exc}")
            QMessageBox.critical(self, "Video Error", str(exc))

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

    def select_batch_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with images")
        if folder:
            self.batch_folder_edit.setText(folder)
            self.input_folder = folder
            self.folder_path_edit.setText(folder)
            self.input_path = ""
            self.image_path_edit.setText("")

    def start_batch_restore(self):
        if not self.input_folder and not self.batch_folder_edit.text():
            QMessageBox.warning(self, "No folder", "Please select a folder first.")
            return

        folder = Path(self.batch_folder_edit.text() or self.input_folder)
        output_folder = Path(self.batch_output_edit.text()) if self.batch_output_edit.text() else folder / "restored"
        output_folder.mkdir(parents=True, exist_ok=True)

        image_files = sorted(list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")))
        if not image_files:
            QMessageBox.warning(self, "No images", "No supported images were found in the selected folder.")
            return

        self.batch_processed_files = []
        self.batch_log.clear()
        self.batch_progress.setValue(0)

        def worker():
            try:
                for idx, img_file in enumerate(image_files):
                    out_file = output_folder / f"{img_file.stem}_restored.png"
                    self.batch_log.append(f"Processing {idx + 1}/{len(image_files)}: {img_file.name}")
                    QApplication.processEvents()

                    ok, out, msg = enhance_image(
                        str(img_file),
                        str(out_file),
                        scale=4 if self.batch_scale_box.currentText() == "4x" else 2,
                        denoise=self.batch_denoise_box.isChecked(),
                        color_boost=1.35,
                        contrast=1.10,
                        sharpness=1.30,
                        face_restore=False,
                        workflow=self.batch_workflow_box.currentText(),
                    )
                    if ok and Path(out).exists():
                        self.batch_processed_files.append(str(out))
                        self.batch_log.append(f"✓ {img_file.name}: done")
                    else:
                        self.batch_log.append(f"✗ {img_file.name}: failed")
                    self.batch_progress.setValue(int((idx + 1) / len(image_files) * 100))
                    QApplication.processEvents()

                self.batch_log.append(f"\nCompleted. Videos can be created from {len(self.batch_processed_files)} processed images.")
                self.create_video_btn.setEnabled(len(self.batch_processed_files) > 0)
                QMessageBox.information(self, "Batch complete", f"Processed {len(self.batch_processed_files)} images in {output_folder}")
            except Exception as exc:
                self.batch_log.append(f"Error: {exc}")
                QMessageBox.critical(self, "Batch Error", str(exc))

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OldPhotoRestorationApp()
    window.show()
    sys.exit(app.exec_())
