import math
from typing import Dict, Tuple
from collections import deque

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String


class DestinationController(Node):
    """중앙 명령을 받아 Nova Carter를 지정된 위치로 이동시키는 노드입니다."""

    def __init__(self) -> None:
        super().__init__("destination_controller")

        # 좌표는 임의 설정 상태!
        self.destinations: Dict[str, Tuple[float, float, float]] = {
            "STORAGE": (-6.421, 9.619, 3.14),
            "DISCARD": (-7.131, -6.657, 3.14),
            "HOME": (-6.006, -1.000, 3.142),
        }

        # 중앙 시스템으로부터 목적지 명령 수신
        self.destination_subscriber = self.create_subscription(
            String,
            "/amr/destination",
            self.destination_callback,
            10,
        )

        # Nav2 NavigateToPose 액션 클라이언트
        self.nav_client = ActionClient(
            self,
            NavigateToPose,
            "/navigate_to_pose",
        )

        self.status_publisher = self.create_publisher(
            String,
            "/amr/status",
            10,
        )

        self.current_status = "IDLE"
        self.publish_status("IDLE")

        self.current_goal_handle = None
        self.is_moving = False
        self.current_destination = None
        self.destination_queue = deque()

        self.get_logger().info(
            "AMR 목적지 제어 노드가 시작되었습니다."
        )
        self.get_logger().info(
            "사용 가능한 명령: STORAGE, DISCARD, HOME"
        )

    def publish_status(self, status: str):
        """현재 AMR 상태를 /amr/status 토픽으로 발행합니다."""

        self.current_status = status

        msg = String()
        msg.data = status
        self.status_publisher.publish(msg)

        self.get_logger().info(
            f"AMR 상태 변경: {status}"
        )

    def destination_callback(self, msg: String) -> None:
        """중앙 시스템에서 목적지 문자열을 받았을 때 실행됩니다."""

        destination_name = msg.data.strip().upper()

        if destination_name not in self.destinations:
            self.get_logger().warning(
                f"등록되지 않은 목적지입니다: {destination_name}"
            )
            return

        if self.is_moving:
            self.destination_queue.append(destination_name)
            self.get_logger().info(
                f"현재 {self.current_destination}로 이동 중입니다. "
                f"{destination_name} 명령을 대기열에 추가합니다."
            )
            self.get_logger().info(
                f"현재 목적지 대기열 : {list(self.destination_queue)}"
            )
            return

        x, y, yaw = self.destinations[destination_name]

        self.send_navigation_goal(
            destination_name=destination_name,
            x=x,
            y=y,
            yaw=yaw,
        )

    def send_navigation_goal(
        self,
        destination_name: str,
        x: float,
        y: float,
        yaw: float,
    ) -> None:
        """Nav2에 목표 위치를 전송합니다."""

        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                "Nav2 NavigateToPose 서버를 찾을 수 없습니다."
            )
            self.get_logger().error(
                "Nav2가 실행 중인지 확인하세요."
            )
            self.publish_status("FAILED")
            return

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()

        # 목적지 좌표는 map 좌표계를 기준으로 작성
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()

        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.position.z = 0.0

        # yaw를 quaternion으로 변환
        goal.pose.pose.orientation.x = 0.0
        goal.pose.pose.orientation.y = 0.0
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.is_moving = True
        self.current_destination = destination_name

        self.get_logger().info(
            f"{destination_name} 이동 명령 전송: "
            f"x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}"
        )

        send_goal_future = self.nav_client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback,
        )

        send_goal_future.add_done_callback(
            self.goal_response_callback
        )

    def goal_response_callback(self, future) -> None:
        """Nav2가 목표를 승인했는지 확인합니다."""

        try:
            goal_handle = future.result()

        except Exception as error:
            self.get_logger().error(
                f"목표 전송 중 오류 발생: {error}"
            )
            self.publish_status("FAILED")
            self.reset_navigation_state()
            return

        if not goal_handle.accepted:
            self.get_logger().error(
                "Nav2가 이동 목표를 거절했습니다."
            )
            self.publish_status("FAILED")
            self.reset_navigation_state()
            return

        self.current_goal_handle = goal_handle

        self.get_logger().info(
            f"{self.current_destination} 이동 목표가 승인되었습니다."
        )
        self.publish_status("MOVING")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            self.navigation_result_callback
        )

    def feedback_callback(self, feedback_msg) -> None:
        """이동 중 남은 거리를 출력합니다."""

        feedback = feedback_msg.feedback
        remaining_distance = feedback.distance_remaining

        self.get_logger().info(
            f"{self.current_destination}까지 "
            f"남은 거리: {remaining_distance:.2f} m",
            throttle_duration_sec=1.0,
        )

    def navigation_result_callback(self, future) -> None:
        """목적지 도착 또는 이동 실패 결과를 처리합니다."""

        try:
            wrapped_result = future.result()

        except Exception as error:
            self.get_logger().error(
                f"Nav2 결과 수신 중 오류 발생: {error}"
            )
            self.publish_status("FAILED")
            self.reset_navigation_state()
            return

        status = wrapped_result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(
                f"{self.current_destination} 도착 완료"
            )
            self.publish_status("ARRIVED")

        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warning(
                f"{self.current_destination} 이동이 취소되었습니다."
            )
            self.publish_status("CANCELED")

        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error(
                f"{self.current_destination} 이동에 실패했습니다."
            )
            self.publish_status("FAILED")

        else:
            self.get_logger().warning(
                f"이동 종료 상태 코드: {status}"
            )
            self.publish_status("FAILED")

        self.reset_navigation_state()
        self.start_next_destination()

    def reset_navigation_state(self) -> None:
        self.is_moving = False
        self.current_destination = None
        self.current_goal_handle = None
    
    def start_next_destination(self) -> None:
        """대기열에 있는 다음 목적지로 이동합니다."""
        if not self.destination_queue:
            return
        next_destination = self.destination_queue.popleft()
        self.get_logger().info(
            f"대기열의 다음 목적지로 이동: {next_destination}"
        )
        x, y, yaw = self.destinations[next_destination]
        self.send_navigation_goal(
            destination_name=next_destination,
            x=x,
            y=y,
            yaw=yaw,
        )

def main(args=None) -> None:
    rclpy.init(args=args)

    node = DestinationController()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info("노드를 종료합니다.")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()