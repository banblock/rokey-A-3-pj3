#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Bool, Int32, String
from std_srvs.srv import Trigger

from recycle_interfaces.msg import PickupList, ShoeInspectionResult
from recycle_interfaces.srv import AmrState


class ControlNodeTest(Node):

    def __init__(self):
        super().__init__("control_node_test")

        self.callback_group = ReentrantCallbackGroup()

        # ==================================================
        # ControlNode가 구독하는 topic에 테스트 데이터 전송
        # ==================================================

        self.vision_result_pub = self.create_publisher(
            ShoeInspectionResult,
            "/vision/inspection_result",
            20,
        )

        self.shoe_trigger_pub = self.create_publisher(
            Bool,
            "/shoe_trigger",
            20,
        )

        # ==================================================
        # ControlNode가 발행하는 topic 수신
        # ==================================================

        self.pickup_sub = self.create_subscription(
            PickupList,
            "/control/pickup",
            self.pickup_callback,
            20,
            callback_group=self.callback_group,
        )

        self.sorter_enable_sub = self.create_subscription(
            Bool,
            "/sorter_a/enable",
            self.sorter_enable_callback,
            20,
            callback_group=self.callback_group,
        )

        self.emergency_stop_sub = self.create_subscription(
            Bool,
            "/control/emergency_stop",
            self.emergency_stop_callback,
            20,
            callback_group=self.callback_group,
        )

        self.start_scene_sub = self.create_subscription(
            Bool,
            "/control/start_scene",
            self.start_scene_callback,
            20,
            callback_group=self.callback_group,
        )

        self.alert_sub = self.create_subscription(
            Int32,
            "/control/alerts",
            self.alert_callback,
            20,
            callback_group=self.callback_group,
        )

        self.items_sub = self.create_subscription(
            String,
            "/control/items",
            self.items_callback,
            20,
            callback_group=self.callback_group,
        )

        # ==================================================
        # ControlNode service client
        # ==================================================

        self.start_client = self.create_client(
            Trigger,
            "/control/start",
            callback_group=self.callback_group,
        )

        self.pause_client = self.create_client(
            Trigger,
            "/control/pause",
            callback_group=self.callback_group,
        )

        self.restart_client = self.create_client(
            Trigger,
            "/control/restart",
            callback_group=self.callback_group,
        )

        self.stop_client = self.create_client(
            Trigger,
            "/control/stop",
            callback_group=self.callback_group,
        )

        self.reset_client = self.create_client(
            Trigger,
            "/control/reset",
            callback_group=self.callback_group,
        )

        self.amr_state_client = self.create_client(
            AmrState,
            "/fms/amr_state",
            callback_group=self.callback_group,
        )

        self.fms_restart_client = self.create_client(
            AmrState,
            "/fms/restart",
            callback_group=self.callback_group,
        )

        # ==================================================
        # 테스트 순서
        # ==================================================

        self.test_step = 0

        self.test_timer = self.create_timer(
            1.0,
            self.run_test,
            callback_group=self.callback_group,
        )

        self.get_logger().info("ControlNodeTest started")

    # ======================================================
    # Topic publish test
    # ======================================================

    def publish_reusable_shoe(self):
        msg = ShoeInspectionResult()
        msg.discard = False
        msg.color = 1
        msg.size = 260

        self.vision_result_pub.publish(msg)

        self.get_logger().info(
            "[SEND] /vision/inspection_result "
            "discard=False, color=1, size=260"
        )

    def publish_discard_shoe(self):
        msg = ShoeInspectionResult()
        msg.discard = True
        msg.color = 2
        msg.size = 280

        self.vision_result_pub.publish(msg)

        self.get_logger().info(
            "[SEND] /vision/inspection_result "
            "discard=True, color=2, size=280"
        )

    def publish_shoe_trigger(self):
        msg = Bool()
        msg.data = True

        self.shoe_trigger_pub.publish(msg)

        self.get_logger().info(
            "[SEND] /shoe_trigger data=True"
        )

    # ======================================================
    # Topic subscribe callback
    # ======================================================

    def pickup_callback(self, msg):
        self.get_logger().info(
            "[RECV] /control/pickup "
            f"place={msg.place}, shoes={list(msg.shoes)}"
        )

    def sorter_enable_callback(self, msg):
        self.get_logger().info(
            "[RECV] /sorter_a/enable "
            f"data={msg.data}"
        )

    def emergency_stop_callback(self, msg):
        self.get_logger().info(
            "[RECV] /control/emergency_stop "
            f"data={msg.data}"
        )

    def start_scene_callback(self, msg):
        self.get_logger().info(
            "[RECV] /control/start_scene "
            f"data={msg.data}"
        )

    def alert_callback(self, msg):
        self.get_logger().info(
            "[RECV] /control/alerts "
            f"code={msg.data}"
        )

    def items_callback(self, msg):
        self.get_logger().info(
            "[RECV] /control/items "
            f"data={msg.data}"
        )

    # ======================================================
    # Service call
    # ======================================================

    def call_trigger_service(self, client, service_name):
        if not client.service_is_ready():
            self.get_logger().warning(
                f"[WAIT] {service_name} service unavailable"
            )
            return False

        request = Trigger.Request()
        future = client.call_async(request)

        future.add_done_callback(
            lambda result_future: self.trigger_response_callback(
                result_future,
                service_name,
            )
        )

        self.get_logger().info(
            f"[CALL] {service_name}"
        )
        return True

    def trigger_response_callback(self, future, service_name):
        try:
            response = future.result()

            self.get_logger().info(
                f"[RESP] {service_name} "
                f"success={response.success}, "
                f"message='{response.message}'"
            )

        except Exception as exc:
            self.get_logger().error(
                f"[ERROR] {service_name}: {exc}"
            )

    def call_amr_state_service(
        self,
        client,
        service_name,
        code,
    ):
        if not client.service_is_ready():
            self.get_logger().warning(
                f"[WAIT] {service_name} service unavailable"
            )
            return False

        request = AmrState.Request()
        request.code = code

        future = client.call_async(request)

        future.add_done_callback(
            lambda result_future: self.amr_state_response_callback(
                result_future,
                service_name,
            )
        )

        self.get_logger().info(
            f"[CALL] {service_name} code={code}"
        )
        return True

    def amr_state_response_callback(
        self,
        future,
        service_name,
    ):
        try:
            response = future.result()

            self.get_logger().info(
                f"[RESP] {service_name} "
            )

        except Exception as exc:
            self.get_logger().error(
                f"[ERROR] {service_name}: {exc}"
            )

    # ======================================================
    # Test sequence
    # ======================================================

    def run_test(self):
        if self.test_step == 0:
            if not self.call_trigger_service(
                self.start_client,
                "/control/start",
            ):
                return

        elif self.test_step == 1:
            self.publish_reusable_shoe()

        elif self.test_step == 2:
            self.publish_shoe_trigger()

        elif self.test_step == 3:
            self.publish_discard_shoe()

        elif self.test_step == 4:
            if not self.call_trigger_service(
                self.pause_client,
                "/control/pause",
            ):
                return

        elif self.test_step == 5:
            if not self.call_trigger_service(
                self.restart_client,
                "/control/restart",
            ):
                return

        elif self.test_step == 6:
            if not self.call_amr_state_service(
                self.amr_state_client,
                "/fms/amr_state",
                1,
            ):
                return

        elif self.test_step == 7:
            if not self.call_amr_state_service(
                self.fms_restart_client,
                "/fms/restart",
                0,
            ):
                return

        elif self.test_step == 8:
            if not self.call_trigger_service(
                self.reset_client,
                "/control/reset",
            ):
                return

        elif self.test_step == 9:
            if not self.call_trigger_service(
                self.stop_client,
                "/control/stop",
            ):
                return

        elif self.test_step == 10:
            self.get_logger().info(
                "All ControlNode communication tests completed"
            )
            self.test_timer.cancel()
            return

        self.test_step += 1


def main(args=None):
    rclpy.init(args=args)

    node = ControlNodeTest()

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