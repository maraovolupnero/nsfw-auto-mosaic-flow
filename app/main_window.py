from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QCloseEvent, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from app.batch_processor import BatchProcessor, ProcessResult
from app.preview import ImagePreview
from app.detector import YoloDetector
from app.pdf_export import create_image_pdf, normalize_pdf_filename
from app.settings import ROOT_DIR, auto_pixel_size, load_settings, parse_class_ids, parse_class_names, save_settings
from app.utils import list_images, read_image, resolve_output_directory, save_image


def _app_icon_path() -> Path:
    bundle_root = Path(getattr(sys, "_MEIPASS", ROOT_DIR))
    return bundle_root / "assets" / "app_icon.png"


APP_STYLE = """
QWidget { background: #08131d; color: #d8e1e8; font-family: "Yu Gothic UI", "Segoe UI"; font-size: 13px; }
QMainWindow { background: #07111a; }
QFrame#panel { background: #0d1a26; border: 1px solid #213140; border-radius: 7px; }
QFrame#setupCard, QFrame#controlCard { background: #111f2c; border: 1px solid #263747; border-radius: 7px; }
QLabel#title { font-size: 23px; font-weight: 700; color: #f1f5f7; }
QLabel#sectionTitle { font-size: 15px; font-weight: 700; color: #f1f5f7; }
QLabel#muted { color: #81919e; }
QLabel#ready { color: #50d2ae; font-weight: 600; }
QLineEdit, QComboBox, QPlainTextEdit { background: #0a1621; border: 1px solid #304253; border-radius: 5px; padding: 7px; selection-background-color: #2cb9a6; }
QLineEdit:focus, QComboBox:focus { border-color: #42ccb7; }
QPushButton { background: #192837; border: 1px solid #334657; border-radius: 5px; padding: 7px 12px; font-weight: 600; }
QPushButton:hover { background: #223548; border-color: #4a6073; }
QPushButton:checked { color: #52d7c2; border-color: #3bc7b1; background: #102d32; }
QPushButton#primary { background: #33bda8; color: #041713; border: none; font-size: 16px; padding: 15px 22px; }
QPushButton#primary:hover { background: #52d2bd; }
QPushButton#danger { background: #1a2939; font-size: 15px; padding: 13px 22px; }
QPushButton:disabled { color: #60717e; background: #111c26; border-color: #22313e; }
QSlider::groove:horizontal { height: 4px; background: #344454; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #3cc9b4; border-radius: 2px; }
QSlider::handle:horizontal { background: #dce4e8; width: 16px; margin: -6px 0; border-radius: 8px; }
QProgressBar { border: none; background: #223140; border-radius: 5px; height: 10px; text-align: center; color: transparent; }
QProgressBar::chunk { background: #42c8a5; border-radius: 5px; }
QTableWidget { background: #0b1722; alternate-background-color: #0f1d29; border: none; gridline-color: #223240; selection-background-color: #153742; }
QHeaderView::section { background: #101f2b; color: #9cadb9; border: none; border-bottom: 1px solid #283a49; padding: 8px; font-weight: 600; }
QTabBar::tab { background: transparent; color: #9baab5; padding: 10px 26px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #4bd6c1; border-bottom-color: #4bd6c1; }
QLabel#imagePreview { background: #050c12; color: #657785; border: none; }
QSplitter::handle { background: #1d2c39; width: 2px; height: 2px; }
QScrollBar:vertical { background: #0b1722; width: 10px; }
QScrollBar::handle:vertical { background: #34495a; border-radius: 5px; min-height: 30px; }
"""


class ProcessorThread(QThread):
    progress = Signal(object)
    log = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, settings: dict) -> None:
        super().__init__()
        self.processor = BatchProcessor(settings, self.progress.emit, self.log.emit)

    def run(self) -> None:
        try:
            self.completed.emit(self.processor.run())
        except Exception as exc:
            self.failed.emit(str(exc))

    def stop(self) -> None:
        self.processor.stop()


class ModelNamesThread(QThread):
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self.model_path = model_path

    def run(self) -> None:
        try:
            detector = YoloDetector(self.model_path, use_gpu_if_available=False)
            self.loaded.emit(detector.names)
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._loading_ui = True
        self.settings = load_settings()
        self.worker: ProcessorThread | None = None
        self.model_loader: ModelNamesThread | None = None
        self.model_names: dict[int, str] = {}
        self.class_checkboxes: dict[int, QCheckBox] = {}
        self.preview_images = {"original": None, "detection": None, "processed": None}
        self.queue_results: dict[int, ProcessResult] = {}
        self.manual_masks: dict[str, np.ndarray] = {}
        self.current_preview_result: ProcessResult | None = None
        self.setWindowTitle("YOLO Mosaic Tool")
        icon_path = _app_icon_path()
        if icon_path.is_file():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1420, 940)
        self.setMinimumSize(820, 600)
        self.setStyleSheet(APP_STYLE)
        self._build_ui()
        self._load_values()
        self._loading_ui = False
        self._refresh_queue()
        if Path(self.settings.get("model_path", "")).is_file():
            self._load_model_names()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 14)
        root.setSpacing(11)

        header = QHBoxLayout()
        title = QLabel("YOLO Mosaic Tool")
        title.setObjectName("title")
        self.subtitle = QLabel("公開前画像のプライバシー処理を、確認しながら安全に")
        self.subtitle.setObjectName("muted")
        header.addWidget(title)
        header.addSpacing(16)
        header.addWidget(self.subtitle)
        header.addStretch()
        root.addLayout(header)

        self.setup_scroll = QScrollArea()
        self.setup_scroll.setWidgetResizable(True)
        self.setup_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.setup_scroll.setMinimumWidth(250)
        self.setup_scroll.setWidget(self._build_setup_panel())
        self.preview_panel = self._build_preview_panel()
        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.settings_scroll.setMinimumWidth(280)
        self.settings_scroll.setWidget(self._build_settings_panel())

        self.top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.top_splitter.addWidget(self.setup_scroll)
        self.top_splitter.addWidget(self.preview_panel)
        self.top_splitter.addWidget(self.settings_scroll)
        self.top_splitter.setSizes([290, 720, 330])
        self.top_splitter.setStretchFactor(1, 1)

        root.addWidget(self.top_splitter, 4)
        root.addWidget(self._build_queue_panel(), 2)
        self.setCentralWidget(central)

    def _panel(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("panel")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        return frame, layout

    def _build_setup_panel(self) -> QFrame:
        frame, layout = self._panel("設定チェックリスト")
        self.model_edit = self._path_card(layout, "モデルファイル", "YOLO .pt", self._choose_model)
        self.input_edit = self._path_card(layout, "入力フォルダ", "画像を含むフォルダ", self._choose_input)
        self.output_edit = self._path_card(layout, "出力フォルダ（任意）", "空欄なら入力フォルダ内へ自動作成", self._choose_output)

        card = QFrame()
        card.setObjectName("setupCard")
        card_layout = QVBoxLayout(card)
        top = QHBoxLayout()
        top.addWidget(QLabel("モザイク対象クラス"))
        top.addStretch()
        self.class_mode = QComboBox()
        self.class_mode.addItem("クラス名で指定", "names")
        self.class_mode.addItem("クラスIDで指定", "ids")
        self.class_mode.currentIndexChanged.connect(self._class_mode_changed)
        top.addWidget(self.class_mode)
        card_layout.addLayout(top)
        self.class_edit = QLineEdit()
        self.class_edit.setPlaceholderText("vagina,penis")
        self.class_edit.editingFinished.connect(self._class_text_changed)
        card_layout.addWidget(self.class_edit)
        hint = QLabel("モデル読み込み後、下の一覧からも選択できます")
        hint.setObjectName("muted")
        card_layout.addWidget(hint)

        self.class_scroll = QScrollArea()
        self.class_scroll.setWidgetResizable(True)
        self.class_scroll.setMinimumHeight(120)
        self.class_scroll.setMaximumHeight(170)
        self.class_scroll.setStyleSheet("QScrollArea { border: 1px solid #263747; border-radius: 5px; }")
        self.class_list_widget = QWidget()
        self.class_list_layout = QVBoxLayout(self.class_list_widget)
        self.class_list_layout.setContentsMargins(8, 6, 8, 6)
        self.class_list_layout.addWidget(QLabel("モデルを選択するとクラス一覧を表示します"))
        self.class_list_layout.addStretch()
        self.class_scroll.setWidget(self.class_list_widget)
        card_layout.addWidget(self.class_scroll)
        layout.addWidget(card)

        ready = QFrame()
        ready.setObjectName("setupCard")
        ready_layout = QHBoxLayout(ready)
        self.ready_badge = QLabel("!")
        self.ready_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ready_layout.addWidget(self.ready_badge)
        ready_text = QVBoxLayout()
        self.ready_label = QLabel("設定を確認してください")
        self.ready_label.setObjectName("ready")
        self.ready_detail = QLabel("モデル、入力フォルダ、対象クラスが必要です")
        self.ready_detail.setObjectName("muted")
        ready_text.addWidget(self.ready_label)
        ready_text.addWidget(self.ready_detail)
        ready_layout.addLayout(ready_text)
        layout.addWidget(ready)
        layout.addStretch()
        note = QLabel("YOLOv8以降の .pt モデルに対応\n処理結果は公開前に目視確認してください")
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(note)
        return frame

    def _path_card(self, parent: QVBoxLayout, title: str, placeholder: str, callback) -> QLineEdit:  # type: ignore[no-untyped-def]
        card = QFrame()
        card.setObjectName("setupCard")
        layout = QVBoxLayout(card)
        top = QHBoxLayout()
        top.addWidget(QLabel(title))
        top.addStretch()
        button = QPushButton("選択")
        button.clicked.connect(callback)
        top.addWidget(button)
        layout.addLayout(top)
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.editingFinished.connect(self._path_edited)
        layout.addWidget(edit)
        parent.addWidget(card)
        return edit

    def _build_preview_panel(self) -> QFrame:
        frame, layout = self._panel("プレビュー")
        self.preview_tabs = QTabBar()
        self.preview_tabs.addTab("Original")
        self.preview_tabs.addTab("Detection")
        self.preview_tabs.addTab("Processed")
        self.preview_tabs.currentChanged.connect(self._change_preview)
        layout.addWidget(self.preview_tabs)
        self.preview = ImagePreview()
        self.preview.wheel_navigate.connect(self._navigate_queue)
        self.preview.mask_changed.connect(self._manual_mask_changed)
        preview_row = QHBoxLayout()
        preview_row.addWidget(self.preview, 1)
        zoom_column = QVBoxLayout()
        zoom_column.addWidget(QLabel("拡大"), alignment=Qt.AlignmentFlag.AlignHCenter)
        self.zoom_slider = QSlider(Qt.Orientation.Vertical)
        self.zoom_slider.setRange(25, 400)
        self.zoom_slider.setSingleStep(5)
        self.zoom_slider.setPageStep(25)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setInvertedAppearance(False)
        self.zoom_slider.valueChanged.connect(self._zoom_changed)
        zoom_column.addWidget(self.zoom_slider, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.zoom_value = QLabel("100%")
        self.zoom_value.setObjectName("ready")
        self.zoom_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        zoom_column.addWidget(self.zoom_value)
        self.zoom_reset_button = QPushButton("全体")
        self.zoom_reset_button.setToolTip("画像全体が収まる100%表示へ戻します")
        self.zoom_reset_button.clicked.connect(self._reset_zoom)
        zoom_column.addWidget(self.zoom_reset_button)
        zoom_column.addWidget(QLabel("縮小"), alignment=Qt.AlignmentFlag.AlignHCenter)
        preview_row.addLayout(zoom_column)
        layout.addLayout(preview_row, 1)
        manual_row = QHBoxLayout()
        manual_label = QLabel("手動追加モザイク")
        manual_label.setObjectName("sectionTitle")
        manual_row.addWidget(manual_label)
        self.brush_button = QPushButton("ブラシ")
        self.brush_button.setCheckable(True)
        self.brush_button.setChecked(True)
        self.brush_button.clicked.connect(lambda: self._set_manual_tool(False))
        manual_row.addWidget(self.brush_button)
        self.eraser_button = QPushButton("消しゴム")
        self.eraser_button.setCheckable(True)
        self.eraser_button.clicked.connect(lambda: self._set_manual_tool(True))
        manual_row.addWidget(self.eraser_button)
        manual_row.addWidget(QLabel("太さ"))
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(5, 200)
        self.brush_slider.setValue(60)
        self.brush_slider.setMaximumWidth(150)
        self.brush_slider.valueChanged.connect(self._brush_size_changed)
        manual_row.addWidget(self.brush_slider)
        self.brush_size_label = QLabel("60px")
        self.brush_size_label.setObjectName("muted")
        manual_row.addWidget(self.brush_size_label)
        self.undo_brush_button = QPushButton("元に戻す")
        self.undo_brush_button.clicked.connect(self.preview.undo)
        manual_row.addWidget(self.undo_brush_button)
        self.clear_brush_button = QPushButton("クリア")
        self.clear_brush_button.clicked.connect(self.preview.clear_mask)
        manual_row.addWidget(self.clear_brush_button)
        self.save_brush_button = QPushButton("追加モザイクを保存")
        self.save_brush_button.clicked.connect(self._save_manual_mosaic)
        manual_row.addWidget(self.save_brush_button)
        layout.addLayout(manual_row)
        self._update_manual_controls(False)
        hint = QLabel("ホイール: 前後画像 / 拡大中の左ドラッグ: 移動 / Processedは右ドラッグで移動")
        hint.setObjectName("muted")
        layout.addWidget(hint)
        footer = QHBoxLayout()
        self.preview_filename = QLabel("ファイル: -")
        self.preview_filename.setObjectName("muted")
        self.preview_count = QLabel("検出数: -")
        self.preview_count.setObjectName("muted")
        footer.addWidget(self.preview_filename)
        footer.addStretch()
        footer.addWidget(self.preview_count)
        layout.addLayout(footer)
        return frame

    def _build_settings_panel(self) -> QFrame:
        frame, layout = self._panel("検出・モザイク設定")
        self.conf_slider, self.conf_value = self._slider_card(layout, "検出閾値 (Confidence)", "この値以上の検出のみを対象にします", 10, 95, 5)
        self.high_recall_check = QCheckBox("高精度検出（遠い・小さい対象向け）")
        self.high_recall_check.setToolTip("画像を重なり付きで分割して追加推論します。検出率は上がりますが処理時間が増えます。")
        self.high_recall_check.toggled.connect(self._save_from_ui)
        layout.addWidget(self.high_recall_check)
        self.auto_check = QCheckBox("画像サイズから自動計算")
        self.auto_check.toggled.connect(self._toggle_auto)
        self.pixel_slider, self.pixel_value = self._slider_card(layout, "モザイク Pixel Size", "大きいほど粗いモザイクになります", 4, 80, 2, self.auto_check)
        self.expand_slider, self.expand_value = self._slider_card(layout, "範囲拡張 (Expand)", "検出範囲の外側に広げます", 0, 50, 5)
        self.feather_slider, self.feather_value = self._slider_card(layout, "境界ぼかし (Feather)", "モザイク境界をなめらかにします", 0, 40, 2)

        self.recommended_button = QPushButton("推奨設定を適用")
        self.recommended_button.setToolTip("このモデル向けの基準値に戻します。適用後も各項目を自由に微調整できます。")
        self.recommended_button.clicked.connect(self._apply_recommended_settings)
        layout.addWidget(self.recommended_button)
        recommended_hint = QLabel("迷った場合の基準値です。適用後に各項目を微調整できます")
        recommended_hint.setObjectName("muted")
        recommended_hint.setWordWrap(True)
        layout.addWidget(recommended_hint)

        pdf_card = QFrame()
        pdf_card.setObjectName("controlCard")
        pdf_layout = QVBoxLayout(pdf_card)
        self.create_pdf_check = QCheckBox("処理完了時にPDFを自動作成")
        self.create_pdf_check.toggled.connect(self._save_from_ui)
        pdf_layout.addWidget(self.create_pdf_check)
        self.pdf_filename_edit = QLineEdit()
        self.pdf_filename_edit.setPlaceholderText("mosaic_images.pdf")
        self.pdf_filename_edit.editingFinished.connect(self._save_from_ui)
        pdf_layout.addWidget(self.pdf_filename_edit)
        pdf_hint = QLabel("画像1枚を1ページとして、出力フォルダ内へ保存します")
        pdf_hint.setObjectName("muted")
        pdf_hint.setWordWrap(True)
        pdf_layout.addWidget(pdf_hint)
        layout.addWidget(pdf_card)
        layout.addStretch()
        return frame

    def _slider_card(self, parent: QVBoxLayout, title: str, hint: str, minimum: int, maximum: int, step: int, extra: QWidget | None = None):  # type: ignore[no-untyped-def]
        card = QFrame()
        card.setObjectName("controlCard")
        layout = QVBoxLayout(card)
        top = QHBoxLayout()
        top.addWidget(QLabel(title))
        top.addStretch()
        value = QLabel("-")
        value.setObjectName("ready")
        top.addWidget(value)
        layout.addLayout(top)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum // step, maximum // step)
        slider.setSingleStep(1)
        slider.setProperty("scale", step)
        slider.valueChanged.connect(self._slider_changed)
        layout.addWidget(slider)
        if extra is not None:
            layout.addWidget(extra)
        description = QLabel(hint)
        description.setObjectName("muted")
        description.setWordWrap(True)
        layout.addWidget(description)
        parent.addWidget(card)
        return slider, value

    def _build_queue_panel(self) -> QFrame:
        frame, layout = self._panel("処理キュー")
        self.queue_table = QTableWidget(0, 5)
        self.queue_table.setHorizontalHeaderLabels(["#", "ファイル名", "検出 / マスク", "ステータス", "出力ファイル"])
        self.queue_table.setAlternatingRowColors(True)
        self.queue_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.queue_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.queue_table.itemSelectionChanged.connect(self._show_selected_queue_image)
        header = self.queue_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        actions = QFrame()
        actions.setMinimumWidth(260)
        action_layout = QGridLayout(actions)
        action_layout.addWidget(QLabel("全体の進捗"), 0, 0)
        self.progress_count = QLabel("0 / 0")
        self.progress_count.setObjectName("muted")
        action_layout.addWidget(self.progress_count, 0, 1, alignment=Qt.AlignmentFlag.AlignRight)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        action_layout.addWidget(self.progress, 1, 0, 1, 2)
        self.current_file = QLabel("待機中")
        self.current_file.setObjectName("muted")
        action_layout.addWidget(self.current_file, 2, 0, 1, 2)
        self.start_button = QPushButton("▶  処理開始")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self._start_processing)
        action_layout.addWidget(self.start_button, 3, 0)
        self.stop_button = QPushButton("■  停止")
        self.stop_button.setObjectName("danger")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_processing)
        action_layout.addWidget(self.stop_button, 3, 1)
        self.create_pdf_button = QPushButton("出力画像からPDFを作成")
        self.create_pdf_button.clicked.connect(self._create_pdf_from_output)
        action_layout.addWidget(self.create_pdf_button, 4, 0, 1, 2)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(1000)
        self.log_view.setPlaceholderText("処理ログ")
        action_layout.addWidget(self.log_view, 5, 0, 1, 2)
        self.queue_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.queue_splitter.addWidget(self.queue_table)
        self.queue_splitter.addWidget(actions)
        self.queue_splitter.setSizes([850, 430])
        self.queue_splitter.setStretchFactor(0, 1)
        layout.addWidget(self.queue_splitter)
        return frame

    def _load_values(self) -> None:
        self.model_edit.setText(self.settings["model_path"])
        self.input_edit.setText(self.settings["input_dir"])
        self.output_edit.setText(self.settings["output_dir"])
        mode = self.settings.get("target_class_mode", "names")
        self.class_mode.setCurrentIndex(1 if mode == "ids" else 0)
        if mode == "ids":
            self.class_edit.setText(",".join(map(str, self.settings.get("target_class_ids", []))))
        else:
            self.class_edit.setText(",".join(self.settings.get("target_class_names", ["vagina", "penis"])))
        self.conf_slider.setValue(round(self.settings["confidence"] * 100 / 5))
        self.high_recall_check.setChecked(self.settings.get("high_recall_detection", True))
        self.pixel_slider.setValue(round(self.settings["pixel_size"] / 2))
        self.expand_slider.setValue(round(self.settings["expand_ratio"] * 100 / 5))
        self.feather_slider.setValue(round(self.settings["feather_px"] / 2))
        self.auto_check.setChecked(self.settings["pixel_size_mode"] != "fixed")
        self.create_pdf_check.setChecked(self.settings.get("create_pdf_after_processing", False))
        self.pdf_filename_edit.setText(self.settings.get("pdf_filename", "mosaic_images.pdf"))
        preview_index = {"original": 0, "detection": 1, "processed": 2}.get(self.settings["preview_mode"], 2)
        self.preview_tabs.setCurrentIndex(preview_index)
        self._update_labels()
        self._update_ready_state()

    def _slider_changed(self) -> None:
        self._update_labels()
        self._save_from_ui()

    def _update_labels(self) -> None:
        self.conf_value.setText(f"{self.conf_slider.value() * 0.05:.2f}")
        self.pixel_value.setText("Auto（長辺÷80）" if self.auto_check.isChecked() else f"{self.pixel_slider.value() * 2}px")
        self.expand_value.setText(f"{self.expand_slider.value() * 5}%")
        self.feather_value.setText(f"{self.feather_slider.value() * 2}px")

    def _toggle_auto(self, checked: bool) -> None:
        self.pixel_slider.setEnabled(not checked)
        self._update_labels()
        self._save_from_ui()

    def _path_edited(self) -> None:
        self._save_from_ui()
        self._refresh_queue()

    def _class_mode_changed(self) -> None:
        mode = self.class_mode.currentData()
        if mode == "ids":
            self.class_edit.setPlaceholderText("2,4,7")
            self.class_edit.setText(",".join(map(str, self.settings.get("target_class_ids", []))))
        else:
            self.class_edit.setPlaceholderText("vagina,penis")
            self.class_edit.setText(",".join(self.settings.get("target_class_names", ["vagina", "penis"])))
        self._save_from_ui()

    def _class_text_changed(self) -> None:
        self._save_from_ui()
        self._sync_checkboxes_from_text()

    def _save_from_ui(self) -> None:
        if self._loading_ui:
            return
        mode = self.class_mode.currentData() or "names"
        class_ids = self.settings.get("target_class_ids", [])
        class_names = self.settings.get("target_class_names", ["vagina", "penis"])
        try:
            if mode == "ids":
                class_ids = parse_class_ids(self.class_edit.text())
            else:
                class_names = parse_class_names(self.class_edit.text())
        except ValueError:
            pass
        self.settings.update({
            "model_path": self.model_edit.text().strip(),
            "input_dir": self.input_edit.text().strip(),
            "output_dir": self.output_edit.text().strip(),
            "target_class_mode": mode,
            "target_class_ids": class_ids,
            "target_class_names": class_names,
            "confidence": self.conf_slider.value() * 0.05,
            "inference_size": 1280,
            "high_recall_detection": self.high_recall_check.isChecked(),
            "pixel_size_mode": "auto_1_80" if self.auto_check.isChecked() else "fixed",
            "pixel_size": self.pixel_slider.value() * 2,
            "expand_ratio": self.expand_slider.value() * 0.05,
            "feather_px": self.feather_slider.value() * 2,
            "create_pdf_after_processing": self.create_pdf_check.isChecked(),
            "pdf_filename": normalize_pdf_filename(self.pdf_filename_edit.text()),
        })
        save_settings(self.settings)
        self._update_ready_state()

    def _update_ready_state(self) -> None:
        model_ok = Path(self.model_edit.text()).is_file() and self.model_edit.text().lower().endswith(".pt")
        input_ok = Path(self.input_edit.text()).is_dir()
        classes_ok = bool(self.class_edit.text().strip())
        if model_ok and input_ok and classes_ok:
            self.ready_badge.setText("✓")
            self.ready_badge.setStyleSheet("background:#36c5a4;color:white;border-radius:18px;font-size:21px;min-width:36px;min-height:36px;")
            self.ready_label.setText("すべての設定が完了しています")
            self.ready_detail.setText("一括処理を開始できます")
        else:
            self.ready_badge.setText("!")
            self.ready_badge.setStyleSheet("background:#344556;color:#c8d2db;border-radius:18px;font-size:19px;min-width:36px;min-height:36px;")
            self.ready_label.setText("設定を確認してください")
            self.ready_detail.setText("モデル、入力フォルダ、対象クラスが必要です")

    def _choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "YOLOモデルを選択", self.model_edit.text(), "YOLO Model (*.pt)")
        if path:
            self.model_edit.setText(path)
            self._save_from_ui()
            self._load_model_names()

    def _load_model_names(self) -> None:
        path = self.model_edit.text().strip()
        if not Path(path).is_file() or self.model_loader is not None:
            return
        self._append_log("モデルのクラス一覧を読み込んでいます...")
        self.model_loader = ModelNamesThread(path)
        self.model_loader.loaded.connect(self._on_model_names_loaded)
        self.model_loader.failed.connect(self._on_model_names_failed)
        self.model_loader.finished.connect(self._model_loader_finished)
        self.model_loader.start()

    def _model_loader_finished(self) -> None:
        self.model_loader = None

    def _on_model_names_loaded(self, names: dict[int, str]) -> None:
        self.model_names = names
        self._populate_class_checkboxes()
        self._append_log(f"モデルクラスを読み込みました: {len(names)}クラス")

    def _on_model_names_failed(self, message: str) -> None:
        self._append_log(f"モデルクラスの読み込みに失敗しました: {message}")

    def _populate_class_checkboxes(self) -> None:
        while self.class_list_layout.count():
            item = self.class_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.class_checkboxes = {}
        selected_ids = set(self.settings.get("target_class_ids", []))
        selected_names = {name.lower() for name in self.settings.get("target_class_names", [])}
        for class_id, name in sorted(self.model_names.items()):
            checkbox = QCheckBox(f"{class_id}: {name}")
            checkbox.setProperty("class_id", class_id)
            checkbox.setProperty("class_name", name)
            checkbox.setChecked(class_id in selected_ids if self.settings.get("target_class_mode") == "ids" else name.lower() in selected_names)
            checkbox.toggled.connect(self._checkbox_selection_changed)
            self.class_checkboxes[class_id] = checkbox
            self.class_list_layout.addWidget(checkbox)
        self.class_list_layout.addStretch()

    def _checkbox_selection_changed(self) -> None:
        selected = [checkbox for checkbox in self.class_checkboxes.values() if checkbox.isChecked()]
        self.settings["target_class_ids"] = [int(checkbox.property("class_id")) for checkbox in selected]
        self.settings["target_class_names"] = [str(checkbox.property("class_name")) for checkbox in selected]
        if self.class_mode.currentData() == "ids":
            self.class_edit.setText(",".join(map(str, self.settings["target_class_ids"])))
        else:
            self.class_edit.setText(",".join(self.settings["target_class_names"]))
        self._save_from_ui()

    def _sync_checkboxes_from_text(self) -> None:
        if not self.class_checkboxes:
            return
        try:
            if self.class_mode.currentData() == "ids":
                selected_ids = set(parse_class_ids(self.class_edit.text()))
                for class_id, checkbox in self.class_checkboxes.items():
                    checkbox.blockSignals(True)
                    checkbox.setChecked(class_id in selected_ids)
                    checkbox.blockSignals(False)
            else:
                selected_names = set(parse_class_names(self.class_edit.text()))
                for checkbox in self.class_checkboxes.values():
                    checkbox.blockSignals(True)
                    checkbox.setChecked(str(checkbox.property("class_name")).lower() in selected_names)
                    checkbox.blockSignals(False)
        except ValueError:
            pass

    def _choose_input(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "入力フォルダを選択", self.input_edit.text())
        if path:
            self.input_edit.setText(path)
            self._save_from_ui()
            self._refresh_queue()

    def _choose_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "出力フォルダを選択", self.output_edit.text())
        if path:
            self.output_edit.setText(path)
            self._save_from_ui()

    def _create_pdf_from_output(self) -> None:
        output_dir = Path(self.output_edit.text().strip())
        if not output_dir.is_dir():
            QMessageBox.warning(self, "出力フォルダなし", "先に画像処理を行うか、出力フォルダを指定してください。")
            return
        image_paths = list_images(output_dir, self.settings.get("supported_extensions", [".png", ".jpg", ".jpeg", ".webp"]))
        if not image_paths:
            QMessageBox.warning(self, "画像なし", "出力フォルダにPDFへまとめる画像がありません。")
            return
        pdf_name = normalize_pdf_filename(self.pdf_filename_edit.text())
        self.pdf_filename_edit.setText(pdf_name)
        self._save_from_ui()
        try:
            pdf_path, page_count = create_image_pdf(image_paths, output_dir / pdf_name)
            self._append_log(f"PDFを作成しました: {pdf_path}（{page_count}ページ）")
            QMessageBox.information(self, "PDF作成完了", f"{page_count}枚の画像をPDFにまとめました。\n{pdf_path}")
        except Exception as exc:
            QMessageBox.critical(self, "PDF作成エラー", f"PDFを作成できませんでした。\n{exc}")

    def _apply_recommended_settings(self) -> None:
        values = self.settings.get("recommended_settings", {})
        self.settings["preset"] = "recommended"
        self.conf_slider.setValue(round(values["confidence"] * 100 / 5))
        self.pixel_slider.setValue(round(values["pixel_size"] / 2))
        self.expand_slider.setValue(round(values["expand_ratio"] * 100 / 5))
        self.feather_slider.setValue(round(values["feather_px"] / 2))
        self.auto_check.setChecked(values["pixel_size_mode"] != "fixed")
        self._save_from_ui()
        self.settings["pixel_size_mode"] = values["pixel_size_mode"]
        save_settings(self.settings)

    def _refresh_queue(self) -> None:
        images = list_images(self.input_edit.text(), self.settings["supported_extensions"])
        self.queue_results = {}
        self.manual_masks = {}
        self.queue_table.setRowCount(len(images))
        for row, path in enumerate(images):
            values = [str(row + 1), path.name, "-", "待機中", "-"]
            for column, value in enumerate(values):
                self.queue_table.setItem(row, column, QTableWidgetItem(value))
        self.progress_count.setText(f"0 / {len(images)}")

    def _validate(self) -> bool:
        try:
            if self.class_mode.currentData() == "ids":
                self.settings["target_class_ids"] = parse_class_ids(self.class_edit.text())
            else:
                self.settings["target_class_names"] = parse_class_names(self.class_edit.text())
        except ValueError as exc:
            QMessageBox.warning(self, "入力エラー", str(exc))
            return False
        if not Path(self.model_edit.text()).is_file():
            QMessageBox.warning(self, "モデル未選択", "YOLOモデル(.pt)を選択してください。")
            return False
        if not Path(self.input_edit.text()).is_dir():
            QMessageBox.warning(self, "入力フォルダ未選択", "入力フォルダを選択してください。")
            return False
        if not list_images(self.input_edit.text(), self.settings["supported_extensions"]):
            QMessageBox.warning(self, "画像なし", "入力フォルダに対応画像がありません。")
            return False
        return True

    def _start_processing(self) -> None:
        self._save_from_ui()
        if not self._validate():
            return
        if not self.output_edit.text().strip():
            output_dir, _ = resolve_output_directory(self.input_edit.text(), "")
            self.output_edit.setText(str(output_dir))
            self.settings["output_dir"] = str(output_dir)
            save_settings(self.settings)
        self.queue_results = {}
        self.manual_masks = {}
        self.log_view.clear()
        self.progress.setValue(0)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.worker = ProcessorThread(dict(self.settings))
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._append_log)
        self.worker.completed.connect(self._on_completed)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _stop_processing(self) -> None:
        if self.worker:
            self.worker.stop()
            self.stop_button.setEnabled(False)
            self.current_file.setText("現在の画像が完了したら停止します...")

    def _on_progress(self, result: ProcessResult) -> None:
        row = result.index - 1
        self.queue_results[row] = result
        if row < self.queue_table.rowCount():
            self.queue_table.setItem(row, 2, QTableWidgetItem(f"{result.detection_count} / {result.masked_count}"))
            status_label = {"mosaic": "完了 (モザイク)", "copy": "完了 (コピー)", "error": "エラー"}.get(result.status, result.status)
            self.queue_table.setItem(row, 3, QTableWidgetItem(status_label))
            self.queue_table.setItem(row, 4, QTableWidgetItem(result.output_path))
            self.queue_table.selectRow(row)
        self.progress.setValue(round(result.index / result.total * 100))
        self.progress_count.setText(f"{result.index} / {result.total}")
        self.current_file.setText(f"処理中: {result.filename}")
        self.preview_filename.setText(f"ファイル: {result.filename}")
        self.preview_count.setText(f"全検出: {result.detection_count} / マスク: {result.masked_count}")
        self._load_result_preview(result)

    def _show_selected_queue_image(self) -> None:
        self._reset_zoom()
        rows = self.queue_table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        result = self.queue_results.get(row)
        if result is not None:
            self._load_result_preview(result)
            return
        filename_item = self.queue_table.item(row, 1)
        if filename_item is None:
            return
        original_path = Path(self.input_edit.text()) / filename_item.text()
        self.preview_images = {"original": read_image(original_path), "detection": None, "processed": None}
        self.current_preview_result = None
        self.preview_filename.setText(f"ファイル: {filename_item.text()}")
        self.preview_count.setText("未処理")
        self._change_preview(self.preview_tabs.currentIndex())

    def _load_result_preview(self, result: ProcessResult) -> None:
        self.current_preview_result = result
        self.preview_images = {
            "original": read_image(result.original_path),
            "detection": read_image(result.detection_preview_path) if result.detection_preview_path else None,
            "processed": read_image(result.output_path),
        }
        self.preview_filename.setText(f"ファイル: {result.filename}")
        self.preview_count.setText(f"全検出: {result.detection_count} / マスク: {result.masked_count}")
        self._change_preview(self.preview_tabs.currentIndex())

    def _change_preview(self, index: int) -> None:
        key = ["original", "detection", "processed"][index]
        self.settings["preview_mode"] = key
        image = self.preview_images.get(key)
        editable = key == "processed" and self.current_preview_result is not None and image is not None
        mask = None
        if editable and self.current_preview_result is not None:
            mask = self.manual_masks.get(self.current_preview_result.output_path)
        pixel_size = self.settings.get("pixel_size", 16)
        if image is not None:
            height, width = image.shape[:2]
            pixel_size = auto_pixel_size(width, height, self.settings.get("pixel_size_mode", "fixed"), pixel_size)
        self.preview.set_image(
            image, mask=mask, editable=editable, pixel_size=pixel_size,
            feather_px=self.settings.get("feather_px", 8),
        )
        self._update_manual_controls(editable)
        save_settings(self.settings)

    def _set_manual_tool(self, erase: bool) -> None:
        self.brush_button.setChecked(not erase)
        self.eraser_button.setChecked(erase)
        self.preview.set_erase(erase)

    def _brush_size_changed(self, value: int) -> None:
        self.brush_size_label.setText(f"{value}px")
        self.preview.set_brush_size(value)

    def _zoom_changed(self, value: int) -> None:
        self.zoom_value.setText(f"{value}%")
        self.preview.set_zoom_percent(value)

    def _reset_zoom(self) -> None:
        self.zoom_slider.setValue(100)
        self.preview.reset_view()

    def _manual_mask_changed(self) -> None:
        if self.current_preview_result is None:
            return
        mask = self.preview.mask()
        if mask is not None:
            self.manual_masks[self.current_preview_result.output_path] = mask
        self.save_brush_button.setEnabled(self.preview.has_mask())

    def _save_manual_mosaic(self) -> None:
        result = self.current_preview_result
        if result is None or not self.preview.has_mask():
            return
        rendered = self.preview.rendered_image()
        if rendered is None:
            return
        try:
            old_output_path = result.output_path
            saved_path, fallback = save_image(result.output_path, rendered)
            if saved_path != Path(result.output_path):
                result.output_path = str(saved_path)
                row = result.index - 1
                self.queue_table.setItem(row, 4, QTableWidgetItem(result.output_path))
            self.preview_images["processed"] = rendered
            self.manual_masks.pop(old_output_path, None)
            row = result.index - 1
            self.queue_table.setItem(row, 3, QTableWidgetItem("完了 (手動補正済)"))
            self._append_log(f"手動追加モザイクを保存しました: {result.filename}" + (f" ({fallback})" if fallback else ""))
            height, width = rendered.shape[:2]
            actual_pixel_size = auto_pixel_size(
                width, height, self.settings.get("pixel_size_mode", "fixed"), self.settings.get("pixel_size", 16)
            )
            self.preview.set_image(
                rendered, editable=True, pixel_size=actual_pixel_size,
                feather_px=self.settings.get("feather_px", 8),
            )
            self.save_brush_button.setEnabled(False)
        except Exception as exc:
            QMessageBox.critical(self, "保存エラー", f"手動モザイクを保存できませんでした。\n{exc}")

    def _navigate_queue(self, direction: int) -> None:
        row_count = self.queue_table.rowCount()
        if row_count == 0:
            return
        rows = self.queue_table.selectionModel().selectedRows()
        current = rows[0].row() if rows else 0
        target = max(0, min(row_count - 1, current + direction))
        if target != current or not rows:
            self.queue_table.selectRow(target)
            self.queue_table.scrollToItem(self.queue_table.item(target, 1))

    def _update_manual_controls(self, enabled: bool) -> None:
        for widget in (
            self.brush_button, self.eraser_button, self.brush_slider,
            self.undo_brush_button, self.clear_brush_button,
        ):
            widget.setEnabled(enabled)
        self.save_brush_button.setEnabled(enabled and self.preview.has_mask())

    def _append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)

    def _on_completed(self, counts: dict) -> None:
        self._set_idle()
        self.current_file.setText("処理が完了しました")
        self._append_log(f"完了: モザイク {counts.get('mosaic', 0)} / コピー {counts.get('copy', 0)} / エラー {counts.get('error', 0)}")

    def _on_failed(self, message: str) -> None:
        self._set_idle()
        self.current_file.setText("処理を開始できませんでした")
        self._append_log(f"エラー: {message}")
        QMessageBox.critical(self, "処理エラー", message)

    def _set_idle(self) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.worker and self.worker.isRunning():
            self.worker.wait(2000)
        self.worker = None

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_from_ui()
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(5000)
        if self.model_loader and self.model_loader.isRunning():
            self.model_loader.wait(5000)
        event.accept()


def run_app() -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("YOLO Mosaic Tool")
    icon_path = _app_icon_path()
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))
    window = MainWindow()
    window.show()
    return app.exec()
