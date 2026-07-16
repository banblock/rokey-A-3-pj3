#!/usr/bin/env python3

import json

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String


class AMRNavigationClient(Node):
    def __init__(self):
        super().__init__('amr_navigation_client')

        # ---- 파라미터: 로봇 식별 + 목적지 좌표 프리셋 ----
        # 실제 환경 구성이 끝나기 전까지는 임시 좌표를 쓰고,
        # 나중에 launch 파일이나 파라미터 파일에서 값만 교체하면 됩니다.
        self.declare_parameter('robot_name', 'carter1')
        self.declare_parameter('zone_a_x', 5.0)
        self.declare_parameter('zone_a_y', 3.0)
        self.declare_parameter('zone_c_x', -5.0)
        self.declare_parameter('zone_c_y', 3.0)

        self.robot_name = self.get_parameter('robot_name').value
        self.destinations = {
            'ZONE_A': (  # 재사용 가능(A/B등급) 보관 장소
                self.get_parameter('zone_a_x').value,
                self.get_parameter('zone_a_y').value,
            ),
            'ZONE_C': (  # 폐기(C등급) 장소
                self.get_parameter('zone_c_x').value,
                self.get_parameter('zone_c_y').value,
            ),
        }

        # ---- Nav2 액션 클라이언트 ----
        self._action_client = ActionClient(
            self,
            NavigateToPose,
            f'/{self.robot_name}/navigate_to_pose',
            callback_group=ReentrantCallbackGroup(),
        )

        # FMS -> AMR: 목적지 할당 명령 구독
        self.destination_sub = self.create_subscription(
            String,
            f'/fms/{self.robot_name}/assign_destination',
            self.destination_callback,
            10,
        )

        # AMR -> FMS: 상태 보고
        self.status_pub = self.create_publisher(
            String, f'/fms/{self.robot_name}/status', 10
        )

        self.busy = False
        self.get_logger().info(f'[{self.robot_name}] 내비게이션 클라이언트 준비 완료')

    # ------------------------------------------------------------------
    def destination_callback(self, msg: String):
        if self.busy:
            self.get_logger().warn(
                f'[{self.robot_name}] 이미 이동 중이라 새 목적지를 무시합니다: {msg.data}'
            )
            return

        try:
            payload = json.loads(msg.data)
            zone = payload['zone']
        except (json.JSONDecodeError, KeyError, TypeError):
            zone = msg.data.strip()

        if zone not in self.destinations:
            self.get_logger().error(f'[{self.robot_name}] 알 수 없는 목적지: {zone}')
            self.publish_status(zone, 'FAILED_UNKNOWN_ZONE')
            return

        x, y = self.destinations[zone]
        self.send_goal(x, y, zone)

    def send_goal(self, x: float, y: float, zone: str):
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f'[{self.robot_name}] Nav2 액션 서버 연결 실패')
            self.publish_status(zone, 'FAILED_SERVER_UNAVAILABLE')
            return

        goal_msg = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0  # 방향은 신경 안 쓰는 단순 케이스
        goal_msg.pose = pose

        self.busy = True
        self.publish_status(zone, 'MOVING')
        self.get_logger().info(f'[{self.robot_name}] {zone}({x}, {y})로 이동 시작')

        send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(
            lambda future: self.goal_response_callback(future, zone)
        )

    def feedback_callback(self, feedback_msg):
        remaining = feedback_msg.feedback.distance_remaining
        self.get_logger().debug(f'[{self.robot_name}] 남은 거리: {remaining:.2f}m')

    def goal_response_callback(self, future, zone: str):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f'[{self.robot_name}] 목표가 거부되었습니다')
            self.busy = False
            self.publish_status(zone, 'FAILED_GOAL_REJECTED')
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda future: self.result_callback(future, zone)
        )

    def result_callback(self, future, zone: str):
        status = future.result().status
        self.busy = False

        # action_msgs/msg/GoalStatus: STATUS_SUCCEEDED = 4
        if status == 4:
            self.get_logger().info(f'[{self.robot_name}] {zone} 도착 완료')
            self.publish_status(zone, 'ARRIVED')
        else:
            self.get_logger().error(f'[{self.robot_name}] {zone} 이동 실패 (status={status})')
            self.publish_status(zone, f'FAILED_STATUS_{status}')

    def publish_status(self, zone: str, state: str):
        msg = String()
        msg.data = json.dumps({'robot': self.robot_name, 'zone': zone, 'state': state})
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AMRNavigationClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()