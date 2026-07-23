#!/usr/bin/env python3

import json
import threading
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
# from db.db_manager import MongoDBManager
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Int32, String, Bool
from recycle_interfaces.msg import PickupList, ShoeInspectionResult


SHOE_TYPES = (
    "shoe_type_1",
    "shoe_type_2",
    "shoe_type_3",
    "shoe_type_4",
)

BATCH_SIZE = 5


class AmrState:
    UNKNOWN = "UNKNOWN"
    READY = "READY"
    RESERVED = "RESERVED"
    TRANSPORTING = "TRANSPORTING"
    RETURNING = "RETURNING"
    DEADLOCK = "DEADLOCK"
    ERROR = "ERROR"



class ControlNode(Node):

    def __init__(self):
        super().__init__("control_node")

        self.callback_group = ReentrantCallbackGroup()
        self.state_lock = threading.RLock()

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
        self.shoe_queues: dict = {
            0 : [],
            1 : [],
            2 : [],
        }

        # --------------------------------------------------
        # MongoDB
        # --------------------------------------------------

        self.db = None #MongoDBManager(reset_on_start=True)

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

        # # FMS 운반 요청 TODO : srv로 변경
        # self.transport_request_pub = self.create_publisher(
        #     String,
        #     "/fms/transport_request",
        #     20,
        # )

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
            f"reusable={reusable}"
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
        with self.state_lock:
            shoe = self.pending_classified_shoes.pop(0)

            if shoe is None:
                self.get_logger().error(
                    f"No vision result for loaded shoe"
                )
                return

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

    # ======================================================
    # 운반 요청 생성
    # ======================================================
    #일단 하나씩 전달
    # def dispatch_available_batches(
    #     self,
    #     shoe_type: int,
    # ) -> None:
    #     """
    #     각 pickup queue에 stack이 5개 이상 쌓이면 해당 정보를 FMS에 전달
    #     """
    #     requests_to_publish = []
    #     with self.state_lock:
    #         for i in range(3):
    #             queue_count = len(self.shoe_queues[shoe_type])

    #             if queue_count < BATCH_SIZE:
    #                 break

    #             shoes = [
    #                 self.shoe_queues[shoe_type].pop(0)
    #                 for _ in range(BATCH_SIZE)
    #             ]


    #             batch = PickupList()
    #             batch.place = shoe_type
    #             batch.shoes = shoes
    #             requests_to_publish.append(batch)
        
    #     if len(requests_to_publish) == 0:
    #         return 

    #     for batch in requests_to_publish:
    #         self._publish_transport_request(batch)

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
    # FMS 상태 처리
    # ======================================================

    # def fms_status_callback(self, msg: String) -> None:
    #     """
    #     FMS 입력 예시:

    #     AMR 준비 완료:
    #     {
    #         "event": "PICKUP_READY",
    #         "amr_id": "shoe_type_1_amr_1",
    #         "shoe_type": "shoe_type_1"
    #     }

    #     요청 수락:
    #     {
    #         "event": "REQUEST_ACCEPTED",
    #         "request_id": "...",
    #         "amr_id": "shoe_type_1_amr_1",
    #         "shoe_type": "shoe_type_1"
    #     }

    #     요청 거절:
    #     {
    #         "event": "REQUEST_REJECTED",
    #         "request_id": "...",
    #         "reason": "no available amr"
    #     }

    #     운송 완료:
    #     {
    #         "event": "TRANSPORT_COMPLETED",
    #         "request_id": "...",
    #         "amr_id": "shoe_type_1_amr_1"
    #     }

    #     교착:
    #     {
    #         "event": "DEADLOCK",
    #         "amr_id": "shoe_type_1_amr_1",
    #         "shoe_type": "shoe_type_1",
    #         "reason": "path blocked"
    #     }
    #     """

    #     try:
    #         data = json.loads(msg.data)
    #         event = str(data["event"])

    #     except (json.JSONDecodeError, KeyError, TypeError) as exc:
    #         self.get_logger().error(
    #             f"Invalid FMS status: {exc}"
    #         )
    #         return

    #     if event == "PICKUP_READY":
    #         self._handle_pickup_ready(data)

    #     elif event == "REQUEST_ACCEPTED":
    #         self._handle_request_accepted(data)

    #     elif event == "REQUEST_REJECTED":
    #         self._handle_request_rejected(data)

    #     elif event == "TRANSPORTING":
    #         self._handle_transporting(data)

    #     elif event == "TRANSPORT_COMPLETED":
    #         self._handle_transport_completed(data)

    #     elif event == "RETURNING":
    #         self._handle_returning(data)

    #     elif event == "DEADLOCK":
    #         self._handle_deadlock(data)

    #     elif event == "ERROR":
    #         self._handle_amr_error(data)

    #     elif event == "RESTARTED":
    #         self._handle_restarted(data)

    #     else:
    #         self.get_logger().warning(
    #             f"Unknown FMS event: {event}"
    #         )

    # def _handle_pickup_ready(
    #     self,
    #     data: dict[str, Any],
    # ) -> None:

    #     amr_id = str(data["amr_id"])
    #     shoe_type = str(data["shoe_type"])

    #     if not self._is_valid_amr(shoe_type, amr_id):
    #         self.get_logger().error(
    #             f"Invalid AMR: {amr_id}"
    #         )
    #         return

    #     with self.state_lock:
    #         self.amr_states[shoe_type][amr_id] = AmrState.READY

    #     self.get_logger().info(
    #         f"AMR pickup ready: {amr_id}"
    #     )

    #     self.dispatch_available_batches(shoe_type)

    # def _handle_request_accepted(
    #     self,
    #     data: dict[str, Any],
    # ) -> None:

    #     request_id = str(data["request_id"])
    #     amr_id = str(data["amr_id"])
    #     shoe_type = str(data["shoe_type"])

    #     with self.state_lock:
    #         batch = self.pending_batches.pop(
    #             request_id,
    #             None,
    #         )

    #         if batch is None:
    #             self.get_logger().warning(
    #                 f"Unknown request accepted: {request_id}"
    #             )
    #             return

    #         if not self._is_valid_amr(shoe_type, amr_id):
    #             self._restore_batch_locked(batch)

    #             self.get_logger().error(
    #                 f"FMS assigned invalid AMR: {amr_id}"
    #             )
    #             return

    #         batch.status = "ACCEPTED"
    #         batch.assigned_amr_id = amr_id

    #         self.active_batches[request_id] = batch

    #         self.amr_states[shoe_type][amr_id] = (
    #             AmrState.TRANSPORTING
    #         )

    #     self._update_batch_status(
    #         request_id,
    #         "ACCEPTED",
    #         amr_id=amr_id,
    #     )

    #     self.get_logger().info(
    #         f"Transport accepted: "
    #         f"request_id={request_id}, "
    #         f"amr={amr_id}"
    #     )

    # def _handle_request_rejected(
    #     self,
    #     data: dict[str, Any],
    # ) -> None:

    #     request_id = str(data["request_id"])
    #     reason = str(data.get("reason", "unknown"))

    #     shoe_type = None

    #     with self.state_lock:
    #         batch = self.pending_batches.pop(
    #             request_id,
    #             None,
    #         )

    #         if batch is None:
    #             self.get_logger().warning(
    #                 f"Unknown request rejected: {request_id}"
    #             )
    #             return

    #         shoe_type = batch.shoe_type
    #         self._restore_batch_locked(batch)

    #     self._update_batch_status(
    #         request_id,
    #         "REJECTED",
    #         reason=reason,
    #     )

    #     self.get_logger().warning(
    #         f"Transport rejected: "
    #         f"request_id={request_id}, "
    #         f"reason={reason}"
    #     )

    #     if shoe_type is not None:
    #         self.dispatch_available_batches(shoe_type)

    # def _handle_transporting(
    #     self,
    #     data: dict[str, Any],
    # ) -> None:

    #     request_id = str(data["request_id"])

    #     with self.state_lock:
    #         batch = self.active_batches.get(request_id)

    #         if batch is None:
    #             return

    #         batch.status = "TRANSPORTING"

    #         if batch.assigned_amr_id is not None:
    #             self.amr_states[
    #                 batch.shoe_type
    #             ][batch.assigned_amr_id] = (
    #                 AmrState.TRANSPORTING
    #             )

    #     self._update_batch_status(
    #         request_id,
    #         "TRANSPORTING",
    #     )

    # def _handle_transport_completed(
    #     self,
    #     data: dict[str, Any],
    # ) -> None:

    #     request_id = str(data["request_id"])
    #     amr_id = str(data["amr_id"])

    #     with self.state_lock:
    #         batch = self.active_batches.pop(
    #             request_id,
    #             None,
    #         )

    #         if batch is None:
    #             self.get_logger().warning(
    #                 f"Unknown completed request: {request_id}"
    #             )
    #             return

    #         self.amr_states[
    #             batch.shoe_type
    #         ][amr_id] = AmrState.RETURNING

    #     self._update_batch_status(
    #         request_id,
    #         "COMPLETED",
    #         amr_id=amr_id,
    #         completed_at=self._now(),
    #     )

    #     self.get_logger().info(
    #         f"Transport completed: "
    #         f"request_id={request_id}, "
    #         f"amr={amr_id}"
    #     )

    # def _handle_returning(
    #     self,
    #     data: dict[str, Any],
    # ) -> None:

    #     amr_id = str(data["amr_id"])
    #     shoe_type = str(data["shoe_type"])

    #     with self.state_lock:
    #         if self._is_valid_amr(shoe_type, amr_id):
    #             self.amr_states[
    #                 shoe_type
    #             ][amr_id] = AmrState.RETURNING

    # def _handle_deadlock(
    #     self,
    #     data: dict[str, Any],
    # ) -> None:

    #     amr_id = str(data["amr_id"])
    #     shoe_type = str(data["shoe_type"])
    #     reason = str(data.get("reason", "unknown"))

    #     with self.state_lock:
    #         if self._is_valid_amr(shoe_type, amr_id):
    #             self.amr_states[
    #                 shoe_type
    #             ][amr_id] = AmrState.DEADLOCK

    #     self._save_event(
    #         {
    #             "event": "DEADLOCK",
    #             "amr_id": amr_id,
    #             "shoe_type": shoe_type,
    #             "reason": reason,
    #             "timestamp": self._now(),
    #         }
    #     )

    #     self.get_logger().error(
    #         f"AMR deadlock: "
    #         f"amr={amr_id}, "
    #         f"reason={reason}"
    #     )

    # def _handle_amr_error(
    #     self,
    #     data: dict[str, Any],
    # ) -> None:

    #     amr_id = str(data["amr_id"])
    #     shoe_type = str(data["shoe_type"])

    #     with self.state_lock:
    #         if self._is_valid_amr(shoe_type, amr_id):
    #             self.amr_states[
    #                 shoe_type
    #             ][amr_id] = AmrState.ERROR

    # def _handle_restarted(
    #     self,
    #     data: dict[str, Any],
    # ) -> None:

    #     amr_id = str(data["amr_id"])
    #     shoe_type = str(data["shoe_type"])

    #     with self.state_lock:
    #         if self._is_valid_amr(shoe_type, amr_id):
    #             self.amr_states[
    #                 shoe_type
    #             ][amr_id] = AmrState.RETURNING

    #     self.get_logger().info(
    #         f"AMR restarted: {amr_id}"
    #     )

    # ======================================================
    # 재시작 요청
    # ======================================================

    # def request_fms_restart(
    #     self,
    #     amr_id: str,
    #     shoe_type: str,
    # ) -> None:
    #     """
    #     운영자가 물리적 교착을 해소한 후 호출할 메서드.
    #     추후 ROS service callback으로 연결할 수 있다.
    #     """

    #     if not self._is_valid_amr(shoe_type, amr_id):
    #         raise ValueError(f"Invalid AMR: {amr_id}")

    #     with self.state_lock:
    #         state = self.amr_states[shoe_type][amr_id]

    #         if state != AmrState.DEADLOCK:
    #             raise RuntimeError(
    #                 f"AMR is not in deadlock state: {state}"
    #             )

    #     request = {
    #         "command": "RESTART",
    #         "amr_id": amr_id,
    #         "shoe_type": shoe_type,
    #         "timestamp": self._now(),
    #     }

    #     msg = String()
    #     msg.data = json.dumps(request)

    #     self.fms_restart_pub.publish(msg)

    #     self.get_logger().warning(
    #         f"FMS restart requested: {amr_id}"
    #     )

    # ======================================================
    # 상태 관리
    # ======================================================

    # def publish_control_status(self) -> None:
    #     with self.state_lock:
    #         status = {
    #             "timestamp": self._now(),
    #             "queue_counts": {
    #                 shoe_type: len(queue)
    #                 for shoe_type, queue
    #                 in self.shoe_queues.items()
    #             },
    #             "pending_vision_count": len(
    #                 self.pending_classified_shoes
    #             ),
    #             "pending_transport_count": len(
    #                 self.pending_batches
    #             ),
    #             "active_transport_count": len(
    #                 self.active_batches
    #             ),
    #             "amr_states": {
    #                 shoe_type: states.copy()
    #                 for shoe_type, states
    #                 in self.amr_states.items()
    #             },
    #         }

    #     msg = String()
    #     msg.data = json.dumps(
    #         status,
    #         ensure_ascii=False,
    #     )

    #     self.control_status_pub.publish(msg)


    # def _restore_batch_locked(
    #     self,
    #     batch: TransportBatch,
    # ) -> None:
    #     """
    #     거절된 요청의 신발을 원래 순서대로 Queue 앞쪽에 복구한다.
    #     """

    #     queue = self.shoe_queues[batch.shoe_type]

    #     for shoe in reversed(batch.shoes):
    #         queue.appendleft(shoe)

        

    # def _is_valid_amr(
    #     self,
    #     shoe_type: str,
    #     amr_id: str,
    # ) -> bool:

    #     if shoe_type not in self.amr_states:
    #         return False

    #     return amr_id in self.amr_states[shoe_type]

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


    # def _update_batch_status(
    #     self,
    #     request_id: str,
    #     status: str,
    #     **fields: Any,
    # ) -> None:

    #     if self.db is None:
    #         return

    #     try:
    #         update_data = {
    #             "status": status,
    #             "updated_at": self._now(),
    #             **fields,
    #         }

    #         self.db.transport_batches.update_one(
    #             {"request_id": request_id},
    #             {"$set": update_data},
    #         )

    #     except Exception as exc:
    #         self.get_logger().error(
    #             f"Failed to update batch: {exc}"
    #         )

    # def _save_event(
    #     self,
    #     event: dict[str, Any],
    # ) -> None:

    #     if self.db is None:
    #         return

    #     try:
    #         self.db.system_events.insert_one(event)

    #     except Exception as exc:
    #         self.get_logger().error(
    #             f"Failed to save event: {exc}"
    #         )

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