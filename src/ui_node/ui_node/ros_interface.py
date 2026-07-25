import json
from dataclasses import dataclass
from typing import Any

import rclpy
from cv_bridge import CvBridge
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int32
from std_srvs.srv import Trigger


# ============================================================
# ROS2 topic / service names
# 실제 프로젝트 인터페이스 이름에 맞게 이 부분만 수정한다.
# ============================================================

VISION_IMAGE_TOPIC = "/d455_TOP/color/image_raw"
MODEL_RESULT_IMAGE_TOPIC = "/vision/result_img1"
MODEL_RESULT_IMAGE_TOPIC2 = "/vision/result_img2"
MODEL_RESULT_IMAGE_TOPIC3 = "/vision/result_img3"

# SYSTEM_STATUS_TOPIC = "/control/status"
SHOE_RESULT_TOPIC = "/vision/inspection_result"
AMR_STATUS_TOPIC = "/fms/amr_status"
AMR_POINT_TOPIC = "/fms/amr_points"
INVENTORY_TOPIC = "/control/items"
ALERT_TOPIC = "/control/alerts"

START_SERVICE = "/control/start"
PAUSE_SERVICE = "/control/pause"
RESTART_SERVICE = "/control/restart"
STOP_SERVICE = "/control/stop"
RESET_SERVICE = "/control/reset"

FMS_RESTART_SERVICE = "/fms/restart"

@dataclass
class ServiceResult:
    command: str
    success: bool
    message: str


class RosSignals(QObject):
    """
    ROS callback 결과를 Qt UI thread로 전달하는 signal 모음.

    이미지 signal:
        np.ndarray 형식의 BGR 이미지 전달

    데이터 signal:
        JSON을 파싱한 dict 또는 list 전달
    """

    vision_image = pyqtSignal(object)
    model_result_image = pyqtSignal(object)

    system_status = pyqtSignal(dict)
    shoe_result = pyqtSignal(dict)

    amr_status = pyqtSignal(list)
    amr_points = pyqtSignal(list)
    inventory = pyqtSignal(list)

    alert = pyqtSignal(int)
    service_result = pyqtSignal(object)


class DashboardRosNode(Node):
    """
    대시보드에서 사용하는 ROS2 통신 전담 노드.

    Subscriber:
        Vision Camera Image
        Model Result Image
        System Status
        Shoe Result
        AMR Status
        AMR Point
        Inventory
        Alert

    Service Client:
        Start
        Pause
        Restart
        Stop
        Reset
    """

    def __init__(self, signals: RosSignals) -> None:
        super().__init__("dashboard_ui_node")

        self.signals = signals
        self.bridge = CvBridge()
        self.callback_group = ReentrantCallbackGroup()

        self._create_subscribers()
        self._create_service_clients()

    def _create_subscribers(self) -> None:
        self.create_subscription(
            Image,
            VISION_IMAGE_TOPIC,
            self._vision_image_callback,
            10,
            callback_group=self.callback_group,
        )

        self.create_subscription(
            Image,
            MODEL_RESULT_IMAGE_TOPIC,
            self._model_result_image_callback,
            10,
            callback_group=self.callback_group,
        )

        # self.create_subscription(
        #     String,
        #     SYSTEM_STATUS_TOPIC,
        #     self._system_status_callback,
        #     10,
        #     callback_group=self.callback_group,
        # )

        self.create_subscription(
            String,
            SHOE_RESULT_TOPIC,
            self._shoe_result_callback,
            10,
            callback_group=self.callback_group,
        )

        self.create_subscription(
            String,
            AMR_STATUS_TOPIC,
            self._amr_status_callback,
            10,
            callback_group=self.callback_group,
        )

        self.create_subscription(
            String,
            AMR_POINT_TOPIC,
            self._amr_point_callback,
            10,
            callback_group=self.callback_group,
        )
    # def _alert_callback(self, msg: String) -> None:
    #     data = self._parse_json(msg.data, ALERT_TOPIC)

    #     if data is not None:
    #         self.signals.alert.emit(data)
        # self.create_subscription(
        #     String,
        #     INVENTORY_TOPIC,
        #     self._inventory_callback,
        #     10,
        #     callback_group=self.callback_group,
        # )

        self.create_subscription(
            Int32,
            ALERT_TOPIC,
            self._alert_callback,
            10,
            callback_group=self.callback_group,
        )

        

    def _create_service_clients(self) -> None:
        self.service_clients = {
            "start": self.create_client(
                Trigger,
                START_SERVICE,
                callback_group=self.callback_group,
            ),
            "pause": self.create_client(
                Trigger,
                PAUSE_SERVICE,
                callback_group=self.callback_group,
            ),
            "restart": self.create_client(
                Trigger,
                RESTART_SERVICE,
                callback_group=self.callback_group,
            ),
            "stop": self.create_client(
                Trigger,
                STOP_SERVICE,
                callback_group=self.callback_group,
            ),
            "reset": self.create_client(
                Trigger,
                RESET_SERVICE,
                callback_group=self.callback_group,
            ),
            "fms_restart": self.create_client(
                Trigger,
                FMS_RESTART_SERVICE,
                callback_group=self.callback_group,
            ),
        }

    # ========================================================
    # Image callbacks
    # ========================================================

    def _vision_image_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8",
        )
        self.signals.vision_image.emit(frame)

    def _model_result_image_callback(self, msg: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8",
        )
        self.signals.model_result_image.emit(frame)

    # ========================================================
    # JSON callbacks
    # ========================================================

    # def _system_status_callback(self, msg: String) -> None:
    #     data = self._parse_json(msg.data, SYSTEM_STATUS_TOPIC)

    #     if data is not None:
    #         self.signals.system_status.emit(data)

    def _shoe_result_callback(self, msg: String) -> None:
        data = self._parse_json(msg.data, SHOE_RESULT_TOPIC)

        if data is not None:
            self.signals.shoe_result.emit(data)

    def _amr_status_callback(self, msg: String) -> None:
        data = self._parse_json(msg.data, AMR_STATUS_TOPIC)

        if data is not None:
            self.signals.amr_status.emit(
                data.get("amrs", [])
            )

    def _amr_point_callback(self, msg: String) -> None:
        data = self._parse_json(msg.data, AMR_POINT_TOPIC)

        if data is not None:
            self.signals.amr_points.emit(
                data.get("amrs", [])
            )

    # def _inventory_callback(self, msg: String) -> None:
    #     data = self._parse_json(msg.data, INVENTORY_TOPIC)

    #     if data is not None:
    #         self.signals.inventory.emit(
    #             data.get("items", [])
    #         )

    def _alert_callback(self, msg: Int32) -> None:
        self.signals.alert.emit(msg.data)

    def _parse_json(
        self,
        raw: str,
        source: str,
    ) -> dict[str, Any] | None:
        try:
            data = json.loads(raw)

        except json.JSONDecodeError as error:
            self.get_logger().error(
                f"Invalid JSON from {source}: {error}"
            )
            return None

        if not isinstance(data, dict):
            self.get_logger().error(
                f"Expected JSON object from {source}"
            )
            return None

        return data

    # ========================================================
    # Service request
    # ========================================================

    def request_command(self, command: str) -> None:
        """
        UI 버튼에서 호출한다.

        지원 명령:
            start
            pause
            restart
            stop
            reset
        """

        client = self.service_clients.get(command)

        if client is None:
            self.signals.service_result.emit(
                ServiceResult(
                    command=command,
                    success=False,
                    message="Unknown command",
                )
            )
            return

        if not client.service_is_ready():
            self.signals.service_result.emit(
                ServiceResult(
                    command=command,
                    success=False,
                    message=f"{command} service unavailable",
                )
            )
            return

        future = client.call_async(Trigger.Request())

        future.add_done_callback(
            lambda result_future: self._service_response_callback(
                command,
                result_future,
            )
        )

    def _service_response_callback(
        self,
        command: str,
        future,
    ) -> None:
        try:
            response = future.result()

            result = ServiceResult(
                command=command,
                success=bool(response.success),
                message=response.message,
            )

        except Exception as error:
            result = ServiceResult(
                command=command,
                success=False,
                message=str(error),
            )

        self.signals.service_result.emit(result)


class RosExecutorThread(QThread):
    """
    rclpy executor를 Qt UI thread와 분리해서 실행한다.

    사용 방법:

        self.ros_thread = RosExecutorThread()
        self.ros_thread.ros_ready.connect(self.on_ros_ready)
        self.ros_thread.start()

    ros_ready payload:

        {
            "node": DashboardRosNode,
            "signals": RosSignals,
        }
    """

    ros_ready = pyqtSignal(object)

    def run(self) -> None:
        rclpy.init(args=None)

        self.signals = RosSignals()
        self.node = DashboardRosNode(self.signals)

        self.executor = MultiThreadedExecutor(
            num_threads=4
        )
        self.executor.add_node(self.node)

        self.ros_ready.emit(
            {
                "node": self.node,
                "signals": self.signals,
            }
        )

        try:
            self.executor.spin()

        finally:
            self.executor.remove_node(self.node)
            self.node.destroy_node()

            if rclpy.ok():
                rclpy.shutdown()

    def stop(self) -> None:
        if hasattr(self, "executor"):
            self.executor.shutdown()


# ============================================================
# Expected JSON payload examples
# ============================================================

"""
SYSTEM_STATUS_TOPIC

{
  "vision": "online",
  "simu lation": "running",
  "fms": "online",
  "controller": "online"
}


SHOE_RESULT_TOPIC

{
  "id": "SHOE-00125",
  "reuse": true,
  "type": "Sneaker",
  "confidence": 0.978,
  "processed_at": "14:31:45"
}


AMR_STATUS_TOPIC

{
  "amrs": [
    {
      "id": "AMR-01",
      "state": "Moving",
      "point": "P03 -> P04",
      "target": "Sneaker"
    }
  ]
}


AMR_POINT_TOPIC

{
  "amrs": [
    {
      "id": "AMR-01",
      "point": 3
    },
    {
      "id": "AMR-02",
      "point": 5
    }
  ]
}


INVENTORY_TOPIC

{
  "items": [
    {
      "id": 125,
      "type": "Sneaker",
      "reuse": true,
      "section": 0,
      "size": 260,
      "amr": "AMR-01",
      "processed_at": "14:31:45"
    }
  ]
}


ALERT_TOPIC

{
  "level": "warning",
  "message": "AMR-02 loading delay",
  "time": "14:31:45"
}
"""

