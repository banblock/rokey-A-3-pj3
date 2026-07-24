
import sys
from datetime import datetime
from typing import Any

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from .ros_interface import (
    DashboardRosNode,
    RosExecutorThread,
    RosSignals,
    ServiceResult,
)


class StatusBadge(QLabel):
    def __init__(
        self,
        text: str,
        status: str = "offline",
    ) -> None:
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedHeight(24)
        self.set_status(text, status)

    def set_status(
        self,
        text: str,
        status: str,
    ) -> None:
        self.setText(text.upper())
        self.setProperty("status", status)
        self.style().unpolish(self)
        self.style().polish(self)


class SectionCard(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("SectionCard")

        title_label = QLabel(title)
        title_label.setObjectName("SectionTitle")

        self.body = QVBoxLayout()
        self.body.setContentsMargins(12, 6, 12, 10)
        self.body.setSpacing(9)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(title_label)
        root.addLayout(self.body)


class ImageView(QLabel):
    def __init__(
        self,
        title: str,
        minimum_height: int,
    ) -> None:
        super().__init__(title)
        self.setObjectName("ImageView")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(minimum_height)
        self.setScaledContents(False)

        self._pixmap: QPixmap | None = None

    def set_cv_image(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape

        image = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()

        self._pixmap = QPixmap.fromImage(image)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._pixmap is None:
            return

        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_pixmap()


class AMRPointGraph(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("AMRGraph")
        self.setMinimumHeight(190)

        self.route_points = [
            (0.08, 0.55),
            (0.23, 0.30),
            (0.42, 0.55),
            (0.60, 0.32),
            (0.78, 0.58),
            (0.92, 0.36),
        ]

        self.amr_points: list[dict[str, Any]] = []

    def set_amr_points(
        self,
        amrs: list[dict[str, Any]],
    ) -> None:
        self.amr_points = amrs
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing
        )

        rect = self.rect().adjusted(28, 36, -28, -28)

        painter.setPen(QPen(QColor("#243044"), 1))

        for index in range(7):
            x = rect.left() + rect.width() * index / 6
            painter.drawLine(
                int(x),
                rect.top(),
                int(x),
                rect.bottom(),
            )

        for index in range(5):
            y = rect.top() + rect.height() * index / 4
            painter.drawLine(
                rect.left(),
                int(y),
                rect.right(),
                int(y),
            )

        pixel_points: list[tuple[int, int]] = []

        for point_x, point_y in self.route_points:
            x = int(rect.left() + rect.width() * point_x)
            y = int(rect.top() + rect.height() * point_y)
            pixel_points.append((x, y))

        painter.setPen(QPen(QColor("#64748B"), 3))

        for start, end in zip(
            pixel_points,
            pixel_points[1:],
        ):
            painter.drawLine(
                start[0],
                start[1],
                end[0],
                end[1],
            )

        for index, (x, y) in enumerate(
            pixel_points,
            start=1,
        ):
            painter.setBrush(QColor("#111827"))
            painter.setPen(QPen(QColor("#94A3B8"), 2))
            painter.drawEllipse(x - 8, y - 8, 16, 16)

            painter.setPen(QColor("#CBD5E1"))
            painter.drawText(
                x - 10,
                y - 17,
                f"P{index}",
            )

        colors = [
            QColor("#22C55E"),
            QColor("#38BDF8"),
            QColor("#F59E0B"),
            QColor("#A78BFA"),
            QColor("#F43F5E"),
            QColor("#14B8A6"),
            QColor("#FB7185"),
            QColor("#84CC16"),
        ]

        for index, amr in enumerate(self.amr_points):
            point_number = int(amr.get("point", 1))
            point_number = max(
                1,
                min(point_number, len(pixel_points)),
            )

            x, y = pixel_points[point_number - 1]
            color = colors[index % len(colors)]
            name = str(
                amr.get(
                    "id",
                    f"AMR-{index + 1:02d}",
                )
            )

            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(
                x - 7,
                y - 7,
                14,
                14,
            )

            painter.setPen(color)
            painter.drawText(
                x + 11,
                y + 5,
                name,
            )


class DashboardWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.ros_node: DashboardRosNode | None = None
        self.ros_signals: RosSignals | None = None

        self.alert_widgets: list[QWidget] = []
        self.alert_empty_label: QLabel | None = None

        self.setWindowTitle(
            "Shoe Return Automation HMI"
        )
        self.resize(1500, 820)

        root = QWidget()
        self.setCentralWidget(root)

        # 헤더 레이아웃 셋팅
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(
            18,
            12,
            18,
            12,
        )
        main_layout.setSpacing(10)

        main_layout.addWidget(self._create_header())

        # 바디 레이아웃 셋팅
        content = QGridLayout()
        content.setHorizontalSpacing(10)
        content.setVerticalSpacing(10)

        content.addWidget(
            self._create_vision_card(),
            0,
            0,
        )
        content.addWidget(
            self._create_amr_graph_card(),
            0,
            1,
        )
        content.addWidget(
            self._create_system_status_card(),
            0,
            2,
        )

        content.addWidget(
            self._create_processed_shoe_card(),
            1,
            0,
        )
        content.addWidget(
            self._create_amr_status_card(),
            1,
            1,
        )
        content.addWidget(
            self._create_alert_card(),
            1,
            2,
        )

        content.addWidget(
            self._create_inventory_card(),
            2,
            0,
            1,
            2,
        )
        content.addWidget(
            self._create_control_card(),
            2,
            2,
        )

        content.setColumnStretch(0, 3)
        content.setColumnStretch(1, 3)
        content.setColumnStretch(2, 2)

        content.setRowStretch(0, 3)
        content.setRowStretch(1, 2)
        content.setRowStretch(2, 3)

        main_layout.addLayout(content)

        self._apply_styles()
        self._start_clock()
        self._start_ros()

    # ========================================================
    # UI construction
    # ========================================================

    def _create_header(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)

        title = QLabel("Shoe Reuse Automation")
        title.setObjectName("AppTitle")

        subtitle = QLabel(
            "Vision · Simulation · FMS · AMR"
        )
        subtitle.setObjectName("AppSubtitle")

        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        self.clock_label = QLabel("--:--:--")
        self.clock_label.setObjectName("ClockLabel")

        self.overall_status_badge = StatusBadge(
            "ROS CONNECTING",
            "warning",
        )
        self.overall_status_badge.setFixedWidth(140)

        layout.addLayout(title_box)
        layout.addStretch()
        layout.addWidget(self.overall_status_badge)
        layout.addSpacing(10)
        layout.addWidget(self.clock_label)

        return widget

    def _create_vision_card(self) -> SectionCard:
        card = SectionCard("Vision Camera")

        self.vision_image_view = ImageView(
            "VISION CAMERA",
            minimum_height=190,
        )

        card.body.addWidget(self.vision_image_view)
        return card

    def _create_amr_graph_card(self) -> SectionCard:
        card = SectionCard("AMR Point Graph")

        self.amr_graph = AMRPointGraph()
        card.body.addWidget(self.amr_graph)

        return card

    def _create_system_status_card(
        self,
    ) -> SectionCard:
        card = SectionCard("System Status")

        self.system_status_badges: dict[
            str,
            StatusBadge,
        ] = {}

        for name in [
            "Vision",
            "Simulation",
            "FMS",
            "Controller",
        ]:
            row = QHBoxLayout()

            label = QLabel(name)
            label.setObjectName("RowLabel")

            badge = StatusBadge(
                "OFFLINE",
                "offline",
            )
            badge.setFixedWidth(92)

            self.system_status_badges[
                name.lower()
            ] = badge

            row.addWidget(label)
            row.addStretch()
            row.addWidget(badge)

            card.body.addLayout(row)

        card.body.addStretch()
        return card

    def _create_processed_shoe_card(
        self,
    ) -> SectionCard:
        card = SectionCard(
            "Processed Shoe / Model Result"
        )

        content = QHBoxLayout()
        content.setSpacing(12)

        self.model_result_view = ImageView(
            "MODEL RESULT",
            minimum_height=160,
        )

        info = QFrame()
        info.setObjectName("InnerPanel")

        info_layout = QVBoxLayout(info)
        info_layout.setContentsMargins(
            12,
            12,
            12,
            12,
        )
        info_layout.setSpacing(9)

        self.shoe_value_labels: dict[
            str,
            QLabel,
        ] = {}

        fields = [
            ("id", "ID"),
            ("reuse", "Reuse"),
            ("type", "Type"),
            ("confidence", "Confidence"),
            ("processed_at", "Processed At"),
        ]

        for key, name in fields:
            row = QHBoxLayout()

            left = QLabel(name)
            left.setObjectName("RowLabel")

            right = QLabel("-")
            right.setObjectName("RowValue")

            self.shoe_value_labels[key] = right

            row.addWidget(left)
            row.addStretch()
            row.addWidget(right)

            info_layout.addLayout(row)

        info_layout.addStretch()

        content.addWidget(
            self.model_result_view,
            3,
        )
        content.addWidget(info, 2)

        card.body.addLayout(content)
        return card

    def _create_amr_status_card(self) -> SectionCard:
        card = SectionCard("AMR Status")

        self.amr_status_layout = card.body

        empty_label = QLabel(
            "AMR status data not received"
        )
        empty_label.setObjectName("MutedText")

        self.amr_status_layout.addWidget(empty_label)

        return card

    def _create_alert_card(self) -> SectionCard:
        card = SectionCard("Errors / Warnings")

        self.alert_layout = card.body

        self.alert_empty_label = QLabel(
            "No errors or warnings"
        )
        self.alert_empty_label.setObjectName(
            "MutedText"
        )

        self.alert_layout.addWidget(
            self.alert_empty_label
        )

        return card

    def _create_inventory_card(self) -> SectionCard:
        card = SectionCard("Inventory Database")

        self.inventory_table = QTableWidget(0, 7)
        self.inventory_table.setHorizontalHeaderLabels(
            [
                "ID",
                "Type",
                "Reuse",
                "Section",
                "Size",
                "AMR",
                "Processed At",
            ]
        )

        self.inventory_table.verticalHeader().setVisible(
            False
        )
        self.inventory_table.setAlternatingRowColors(True)
        self.inventory_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.inventory_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.inventory_table.setShowGrid(False)

        header = self.inventory_table.horizontalHeader()

        for column in range(7):
            header.setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.Stretch,
            )

        card.body.addWidget(self.inventory_table)
        return card

    def _create_control_card(self) -> SectionCard:
        card = SectionCard("Manual Control")

        self.control_buttons: dict[
            str,
            QPushButton,
        ] = {}

        commands = [
            ("start", "START", "primary"),
            ("pause", "PAUSE", "warning"),
            ("restart", "RESTART", "secondary"),
            ("stop", "STOP", "danger"),
            ("reset", "RESET", "secondary"),
        ]

        for command, text, kind in commands:
            button = QPushButton(text)
            button.setProperty("kind", kind)
            button.setMinimumHeight(40)
            button.setEnabled(False)
            button.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
            button.clicked.connect(
                lambda checked=False, name=command:
                    self._send_command(name)
            )

            self.control_buttons[command] = button
            card.body.addWidget(button)

        card.body.addStretch()
        return card

    # ========================================================
    # ROS connection
    # ========================================================

    def _start_ros(self) -> None:
        self.ros_thread = RosExecutorThread()
        self.ros_thread.ros_ready.connect(
            self._on_ros_ready
        )
        self.ros_thread.start()

    def _on_ros_ready(
        self,
        payload: dict[str, Any],
    ) -> None:
        self.ros_node = payload["node"]
        self.ros_signals = payload["signals"]

        self.ros_signals.vision_image.connect(
            self.vision_image_view.set_cv_image
        )
        self.ros_signals.model_result_image.connect(
            self.model_result_view.set_cv_image
        )
        # self.ros_signals.system_status.connect(
        #     self._update_system_status
        # )
        self.ros_signals.shoe_result.connect(
            self._update_shoe_result
        )
        self.ros_signals.amr_status.connect(
            self._update_amr_status
        )
        self.ros_signals.amr_points.connect(
            self.amr_graph.set_amr_points
        )
        self.ros_signals.inventory.connect(
            self._update_inventory
        )
        self.ros_signals.alert.connect(
            self._add_alert
        )
        self.ros_signals.service_result.connect(
            self._handle_service_result
        )

        for button in self.control_buttons.values():
            button.setEnabled(True)

        self.overall_status_badge.set_status(
            "ROS CONNECTED",
            "ok",
        )

    # ========================================================
    # ROS data -> UI
    # ========================================================

    def _update_system_status(
        self,
        data: dict[str, Any],
    ) -> None:
        online_count = 0

        for name, badge in (
            self.system_status_badges.items()
        ):
            value = str(
                data.get(name, "offline")
            )
            style = self._map_status_style(value)

            badge.set_status(value, style)

            if style == "ok":
                online_count += 1

        if online_count == len(
            self.system_status_badges
        ):
            self.overall_status_badge.set_status(
                "SYSTEM RUNNING",
                "ok",
            )

        elif online_count == 0:
            self.overall_status_badge.set_status(
                "SYSTEM OFFLINE",
                "offline",
            )

        else:
            self.overall_status_badge.set_status(
                "SYSTEM DEGRADED",
                "warning",
            )

    def _update_shoe_result(
        self,
        data: dict[str, Any],
    ) -> None:
        reuse_value = data.get("reuse", "-")

        if isinstance(reuse_value, bool):
            reuse_text = (
                "PASS"
                if reuse_value
                else "REJECT"
            )
        else:
            reuse_text = str(reuse_value)

        confidence = data.get(
            "confidence",
            "-",
        )

        if isinstance(confidence, (int, float)):
            if confidence <= 1.0:
                confidence_text = (
                    f"{confidence * 100:.1f}%"
                )
            else:
                confidence_text = (
                    f"{confidence:.1f}%"
                )
        else:
            confidence_text = str(confidence)

        values = {
            "id": data.get("id", "-"),
            "reuse": reuse_text,
            "type": data.get("type", "-"),
            "confidence": confidence_text,
            "processed_at": data.get(
                "processed_at",
                "-",
            ),
        }

        for key, value in values.items():
            self.shoe_value_labels[key].setText(
                str(value)
            )

    def _update_amr_status(
        self,
        amrs: list[dict[str, Any]],
    ) -> None:
        self._clear_layout(self.amr_status_layout)

        if not amrs:
            empty_label = QLabel("No AMR status")
            empty_label.setObjectName("MutedText")
            self.amr_status_layout.addWidget(
                empty_label
            )
            return

        for amr in amrs:
            frame = QFrame()
            frame.setObjectName("AMRRow")

            row = QGridLayout(frame)
            row.setContentsMargins(
                10,
                8,
                10,
                8,
            )
            row.setHorizontalSpacing(8)

            amr_id = str(amr.get("id", "-"))
            state = str(
                amr.get("state", "Idle")
            )
            point = str(amr.get("point", "-"))
            target = str(
                amr.get("target", "-")
            )

            name_label = QLabel(amr_id)
            name_label.setObjectName("RowValue")

            badge = StatusBadge(
                state,
                self._map_amr_state_style(state),
            )
            badge.setFixedWidth(78)

            point_label = QLabel(point)
            point_label.setObjectName("MutedText")

            target_label = QLabel(target)
            target_label.setObjectName("MutedText")

            row.addWidget(name_label, 0, 0)
            row.addWidget(badge, 0, 1)
            row.addWidget(point_label, 0, 2)
            row.addWidget(target_label, 0, 3)
            row.setColumnStretch(2, 2)

            self.amr_status_layout.addWidget(frame)

        self.amr_status_layout.addStretch()

    def _update_inventory(
        self,
        items: list[dict[str, Any]],
    ) -> None:
        self.inventory_table.setRowCount(
            len(items)
        )

        for row_index, item_data in enumerate(
            items
        ):
            reuse_value = item_data.get(
                "reuse",
                "-",
            )

            if isinstance(reuse_value, bool):
                reuse_text = (
                    "PASS"
                    if reuse_value
                    else "REJECT"
                )
            else:
                reuse_text = str(reuse_value)

            values = [
                item_data.get("id", "-"),
                item_data.get("type", "-"),
                reuse_text,
                item_data.get("section", "-"),
                item_data.get("size", "-"),
                item_data.get("amr", "-"),
                item_data.get(
                    "processed_at",
                    "-",
                ),
            ]

            for column_index, value in enumerate(
                values
            ):
                table_item = QTableWidgetItem(
                    str(value)
                )
                table_item.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter
                )

                self.inventory_table.setItem(
                    row_index,
                    column_index,
                    table_item,
                )

    def _add_alert(
        self,
        error_code: int,
    ) -> None:

        if error_code != 0:
            return

        message = (
            "[ERROR 000]\n\n"
            "FMS 교착 발생\n\n"
            "교착 상태를 해결한 후\n"
            "FMS를 재시작하십시오."
        )

        QMessageBox.critical(
            self,
            "System Error",
            message,
        )

        reply = QMessageBox.question(
            self,
            "FMS Restart",
            "교착 상태를 해결했습니까?\n"
            "FMS를 재시작하시겠습니까?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.ros_node.request_command(
                "fms_restart"
            )

    # ========================================================
    # UI button -> ROS service
    # ========================================================

    def _send_command(self, command: str) -> None:
        if self.ros_node is None:
            QMessageBox.warning(
                self,
                "ROS",
                "ROS node is not ready",
            )
            return

        button = self.control_buttons[command]
        button.setEnabled(False)

        self.ros_node.request_command(command)

    def _handle_service_result(
        self,
        result: ServiceResult,
    ) -> None:
        if not result.success:

            QMessageBox.warning(
                self,
                result.command.upper(),
                result.message,
            )

    # ========================================================
    # Utility
    # ========================================================

    @staticmethod
    def _clear_layout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()

            if widget is not None:
                widget.deleteLater()

            elif child_layout is not None:
                DashboardWindow._clear_layout(
                    child_layout
                )

    @staticmethod
    def _map_status_style(value: str) -> str:
        normalized = value.strip().lower()

        if normalized in {
            "online",
            "running",
            "connected",
            "ready",
            "active",
            "normal",
        }:
            return "ok"

        if normalized in {
            "warning",
            "delayed",
            "paused",
            "degraded",
            "busy",
        }:
            return "warning"

        return "offline"

    @staticmethod
    def _map_amr_state_style(
        state: str,
    ) -> str:
        normalized = state.strip().lower()

        if normalized == "moving":
            return "ok"

        if normalized == "loading":
            return "warning"

        if normalized in {
            "idle",
            "returning",
        }:
            return "info"

        return "offline"

    @staticmethod
    def _map_alert_style(level: str) -> str:
        normalized = level.strip().lower()

        if normalized in {
            "error",
            "critical",
            "fatal",
        }:
            return "error"

        if normalized in {
            "warning",
            "warn",
        }:
            return "warning"

        if normalized in {
            "normal",
            "ok",
            "success",
        }:
            return "ok"

        return "info"

    def _start_clock(self) -> None:
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(
            self._update_clock
        )
        self.clock_timer.start(1000)

        self._update_clock()

    def _update_clock(self) -> None:
        self.clock_label.setText(
            datetime.now().strftime("%H:%M:%S")
        )

    def closeEvent(self, event) -> None:
        if hasattr(self, "ros_thread"):
            self.ros_thread.stop()
            self.ros_thread.wait(3000)

        event.accept()

    # ========================================================
    # Style
    # ========================================================

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            * {
                font-family: "Inter", "Noto Sans KR", "Arial";
                color: #E5E7EB;
            }

            QMainWindow,
            QWidget {
                background: #0B1220;
            }

            #AppTitle {
                font-size: 25px;
                font-weight: 700;
                color: #F8FAFC;
            }

            #AppSubtitle {
                font-size: 12px;
                color: #94A3B8;
            }

            #ClockLabel {
                font-size: 14px;
                font-weight: 600;
                color: #CBD5E1;
            }

            #SectionCard {
                background: #111827;
                border: 1px solid #1F2937;
                border-radius: 13px;
            }

            #SectionTitle {
                padding: 10px 12px 6px 12px;
                font-size: 14px;
                font-weight: 700;
                color: #F8FAFC;
            }

            #ImageView,
            #AMRGraph {
                background: #090E18;
                border: 1px solid #243244;
                border-radius: 10px;
            }

            #ImageView {
                color: #94A3B8;
                font-size: 14px;
                font-weight: 700;
            }

            #InnerPanel,
            #AMRRow {
                background: #0F172A;
                border: 1px solid #243244;
                border-radius: 9px;
            }

            #RowLabel {
                color: #94A3B8;
                font-size: 12px;
            }

            #RowValue {
                color: #F8FAFC;
                font-size: 12px;
                font-weight: 700;
            }

            #MutedText {
                color: #94A3B8;
                font-size: 11px;
            }

            QLabel[status="ok"] {
                background: rgba(34, 197, 94, 0.14);
                color: #4ADE80;
                border: 1px solid rgba(34, 197, 94, 0.35);
                border-radius: 12px;
                font-size: 10px;
                font-weight: 700;
                padding: 0 8px;
            }

            QLabel[status="warning"] {
                background: rgba(245, 158, 11, 0.14);
                color: #FBBF24;
                border: 1px solid rgba(245, 158, 11, 0.35);
                border-radius: 12px;
                font-size: 10px;
                font-weight: 700;
                padding: 0 8px;
            }

            QLabel[status="info"] {
                background: rgba(56, 189, 248, 0.14);
                color: #38BDF8;
                border: 1px solid rgba(56, 189, 248, 0.35);
                border-radius: 12px;
                font-size: 10px;
                font-weight: 700;
                padding: 0 8px;
            }

            QLabel[status="error"] {
                background: rgba(239, 68, 68, 0.14);
                color: #F87171;
                border: 1px solid rgba(239, 68, 68, 0.35);
                border-radius: 12px;
                font-size: 10px;
                font-weight: 700;
                padding: 0 8px;
            }

            QLabel[status="offline"] {
                background: rgba(100, 116, 139, 0.14);
                color: #94A3B8;
                border: 1px solid rgba(100, 116, 139, 0.35);
                border-radius: 12px;
                font-size: 10px;
                font-weight: 700;
                padding: 0 8px;
            }

            QTableWidget {
                background: #0F172A;
                alternate-background-color: #111827;
                border: 1px solid #243244;
                border-radius: 9px;
                gridline-color: transparent;
                selection-background-color: #1E3A5F;
            }

            QHeaderView::section {
                background: #162033;
                color: #94A3B8;
                border: none;
                padding: 9px;
                font-size: 11px;
                font-weight: 700;
            }

            QPushButton {
                border-radius: 8px;
                padding: 0 12px;
                font-size: 11px;
                font-weight: 700;
            }

            QPushButton:disabled {
                background: #111827;
                border: 1px solid #1F2937;
                color: #475569;
            }

            QPushButton[kind="primary"] {
                background: #2563EB;
                border: 1px solid #3B82F6;
                color: white;
            }

            QPushButton[kind="warning"] {
                background: #78350F;
                border: 1px solid #B45309;
                color: #FEF3C7;
            }

            QPushButton[kind="danger"] {
                background: #7F1D1D;
                border: 1px solid #B91C1C;
                color: #FEE2E2;
            }

            QPushButton[kind="secondary"] {
                background: #182235;
                border: 1px solid #334155;
                color: #CBD5E1;
            }
            """
        )


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = DashboardWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
