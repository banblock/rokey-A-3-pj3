#!/usr/bin/env python3

import json
import threading
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from .db.db_manager import MongoDBManager
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Int32, String, Bool
from std_srvs.srv import Trigger
from recycle_interfaces.msg import PickupList, ShoeInspectionResult
from recycle_interfaces.srv import AmrState


# SHOE_TYPES = (
#     "shoe_type_1",
#     "shoe_type_2",
#     "shoe_type_3",
#     "shoe_type_4",
# )

BATCH_SIZE = 5

START_SERVICE = "/control/start"
PAUSE_SERVICE = "/control/pause"
RESTART_SERVICE = "/control/restart"
STOP_SERVICE = "/control/stop"
RESET_SERVICE = "/control/reset"




class ControlNode(Node):

    def __init__(self):
        super().__init__("control_node")

        self.callback_group = ReentrantCallbackGroup()
        self.state_lock = threading.RLock()

        self.factory_state = False
        self.factory_pause = False

        self.amr_state = True

        self.declare_parameter(
            "mongodb_uri",
            "mongodb://localhost:27018",
        )
        self.declare_parameter(
            "database_name",
            "inventory_db",
        )

        self.mongodb_uri = (
            self.get_parameter("mongodb_uri")
            .get_parameter_value()
            .string_value
        )

        self.database_name = (
            self.get_parameter("database_name")
            .get_parameter_value()
            .string_value
        )

        # --------------------------------------------------
        # 내부 상태
        # --------------------------------------------------

        # Vision 판정은 끝났지만 아직 적재 완료 센서가 오지 않은 신발
        self.pending_classified_shoes: list[list] = []

        # 종류별 적재 완료 신발 Queue
        # self.shoe_queues: dict = {
        #     0 : [],
        #     1 : [],
        #     2 : [],
        # }

        # --------------------------------------------------
        # MongoDB
        # --------------------------------------------------

        self.db = MongoDBManager(
            uri=self.mongodb_uri,
            database_name=self.database_name,
            reset_on_start=True,
        )

        # --------------------------------------------------
        # Subscribers
        # --------------------------------------------------

        # Vision 결과 수신 TODO: 인터페이스 수정
        self.vision_result_sub = self.create_subscription(
            ShoeInspectionResult,
            "/vision/inspection_result",
            self.vision_result_callback,
            20,
            callback_group=self.callback_group,
        )

        # 컨베이어 분류 구역 적재 완료 센서
        self.loading_complete_sub = self.create_subscription(
            Bool,
            "/shoe_trigger",
            self.loading_complete_callback,
            20,
            callback_group=self.callback_group,
        )
        # FMS 및 AMR 상태
        # self.fms_status_sub = self.create_subscription(
        #     String,
        #     "/fms/status",
        #     self.fms_status_callback,
        #     50,
        #     callback_group=self.callback_group,
        # )

        # --------------------------------------------------
        # Publishers
        # --------------------------------------------------

        # FMS 운송 정보 제공
        self.transport_request_pub = self.create_publisher(
            PickupList,
            "/control/pickup",
            20
        )

        # 컨베이어 분류기 제어
        self.conveyor_command_pub = self.create_publisher(
            Bool,
            "/sorter_a/enable",
            20,
        )

        self.conveyor_stop_pub = self.create_publisher(
            Bool,
            "/control/emergency_stop",
            20
        )

        self.factory_start_pub = self.create_publisher(
            Bool,
            "/control/start_scene",
            20
        )

        self.alter_pub = self.create_publisher(
            Int32,
            "/control/alerts",
            20
        )

        #db
        self.inventory_ui_pub = self.create_publisher(
            String,
            "/control/items",
            20,
        )


        # --------------------------------------------------
        # Service Client
        # --------------------------------------------------

        
        self.fms_restart_client = self.create_client(
            Trigger,
            "/control/amr_restart"
        )

        # --------------------------------------------------
        # Service Server
        # --------------------------------------------------

        self.ui_start_server = self.create_service(
            Trigger,
            START_SERVICE,
            self.start_callback,
        )

        self.ui_stop_server = self.create_service(
            Trigger,
            STOP_SERVICE,
            self.stop_callback,
        )

        self.ui_pause_server = self.create_service(
            Trigger,
            PAUSE_SERVICE,
            self.pause_callback,
        )

        self.ui_restart_server = self.create_service(
            Trigger,
            RESTART_SERVICE,
            self.restart_callback,
        )

        self.ui_reset_server = self.create_service(
            Trigger,
            RESET_SERVICE,
            self.reset_callback,
        )

        self.fms_worning_server = self.create_service(
            AmrState,
            "/fms/amr_state",
            self.worning_callback,
        )

        self.fms_restart_server = self.create_service(
            AmrState,
            "/fms/restart",
            self.fms_restart_callback,
        )
        

        # # 교착 해소 후 FMS 재시작 요청
        # self.fms_restart_pub = self.create_publisher(
        #     String,
        #     "/fms/restart_request",
        #     10,
        # )

        # 중앙 노드 상태 출력
        # self.control_status_pub = self.create_publisher(
        #     String,
        #     "/control/status",
        #     10,
        # )

        # 상태 주기 출력
        # self.status_timer = self.create_timer(
        #     2.0,
        #     self.publish_control_status,
        #     callback_group=self.callback_group,
        # )

        self.get_logger().info("ControlNode started")

    # ======================================================
    # Vision 처리
    # ======================================================

    def vision_result_callback(self, msg):
        """vision 결과 callback"""
        try:
            reusable = msg.discard
            shoe = [msg.color, msg.size]

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().error(
                f"Invalid vision result: {exc}"
            )
            return

        if not reusable:
            with self.state_lock:
                self.pending_classified_shoes.append(shoe) 

            self._save_reusable_shoe(shoe)

        self._publish_conveyor_sort_command(reusable, msg.color)

        self.get_logger().info(
            f"Vision result received: "
            f"id={msg.color}, "
            f"type={msg.size}, "
            f"discard={reusable}"
        )

    def _publish_conveyor_sort_command(
        self,
        reusable,
        shoe,
    ):

        # if reusable:
        #     destination = shoe[0]
        # else:
        #     destination = "reject"

        msg = Bool()
        msg.data = not reusable

        self.conveyor_command_pub.publish(msg)
        
    # ======================================================
    # 적재 완료 처리
    # ======================================================

    def loading_complete_callback(self, msg) -> None:
        """
        Vision 결과가 아니라 이 신호가 수신된 시점에
        종류별 Queue에 신발을 추가한다.
        """
        if not msg.data:
            return


        with self.state_lock:
            if len(self.pending_classified_shoes) == 0:
                self.get_logger().error(
                    f"No vision result for loaded shoe"
                )
                return
            
            shoe = self.pending_classified_shoes.pop(0)

            # self.shoe_queues[shoe[0]].append(shoe)

            # queue_count = len(
            #     self.shoe_queues[shoe[0]]
            # )

        self.get_logger().info(
            f"Added to queue: "
            f"type={shoe[0]}, "
            f"shoe_size={shoe[1]}, "
            # f"count={queue_count}"
        )

        batch = PickupList()
        batch.place = shoe[0]
        batch.shoes = [shoe[1]]
        self._publish_transport_request(batch)


    def _publish_transport_request(
        self,
        batch: PickupList,
    ) -> None:
        """
        중앙 노드는 운반할 섹션과 운반할 신발 정보만 전달
        """

        self.transport_request_pub.publish(batch)

        self.get_logger().info(
            f"Transport requested: "
            f"request_id={batch.place}, "
            f"count={len(batch.shoes)}"
        )

    
    # ======================================================
    # MongoDB
    # ======================================================

    def _save_reusable_shoe(
        self,
        shoe,
    ) -> None:

        if self.db is None:
            return

        try:
            self.db.create_item(
                section=shoe[0],
                size=shoe[1]
            )

        except Exception as exc:
            self.get_logger().error(
                f"Failed to save shoe: {exc}"
            )

        self.publish_inventory_to_ui()


        

    def publish_inventory_to_ui(self) -> None:
        try:
            items = self.db.get_items()

            payload = {
                "items": [
                    item.to_dict()
                    for item in items
                ],
                "count": len(items),
            }

            msg = String()
            msg.data = json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            )

            self.inventory_ui_pub.publish(msg)

        except Exception as exc:
            self.get_logger().error(
                f"Failed to publish inventory data: {exc}"
            )
        
    # ======================================================
    # service callback
    # ======================================================
    def start_callback(self, request, response):

        if self.factory_state:
            response.success = False
            response.message = "already start task"
            return response
        
        self.factory_start_pub.publish(Bool(data=True))
        self.factory_state = True
        response.success = True
        response.message = "start task"

        return response

    def stop_callback(self, request, response):
        if not self.factory_state:
            response.success = False
            response.message = "already stop task"
            return response
            
        self.factory_start_pub.publish(Bool(data=False))
        self.factory_state = False
        response.success = True
        response.message = "stop task"
        return response

    def pause_callback(self, request, response):
    
        if self.factory_pause:
            response.success = False
            response.message = "already pause task"
            return response
        
        self.conveyor_stop_pub.publish(Bool(data=True))
        self.factory_pause = True
        response.success = True
        response.message = "pause task"
        return response

    def restart_callback(self, request, response):
        if not self.factory_pause:
            response.success = False
            response.message = "no pause task"
            return response
        
        self.conveyor_stop_pub.publish(Bool(data=False))
        self.factory_pause = False
        response.success = True
        response.message = "restart task"
        return response

    def reset_callback(self, request, response):
            response.success = True
            response.message = "reset task"
            return response

    def worning_callback(self, request, response):
        code = request.code
        self.alter_pub.publish(Int32(data=code))
        self.amr_state = False
        return response

    def fms_restart_callback(self, request, response):
        if self.amr_state:
            response.success = False
            response.message = "None worning"
            return response

        if not self.fms_restart_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning("Start service is not available.")
            response.success = False
            response.message = "Start service is not available"
            return response

        control_request = Trigger.Request()
        future = self.fms_restart_client.call_async(control_request)
        future.add_done_callback(self.fms_restart_response_callback)

        response.success = True
        response.message = "restart amr"
        return response
        
    def fms_restart_response_callback(self, future):
        response = future.result()

        if response.success:
            self.get_logger().info("Restart success")
            self.amr_state = True
        else:
            self.get_logger().error(response.message)
            self.amr_state = False
            self.alter_pub.publish(Int32(data=1))

    # ======================================================
    # 공통
    # ======================================================

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def destroy_node(self) -> bool:

        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)

    node = ControlNode()

    executor = MultiThreadedExecutor(
        num_threads=4,
    )

    executor.add_node(node)

    try:
        executor.spin()

    except KeyboardInterrupt:
        pass

    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()