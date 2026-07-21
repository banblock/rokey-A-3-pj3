#!/usr/bin/env python3
"""
Vision 판정 노드.

동작 방식:
  평소 카메라 콜백은 최신 프레임만 갱신하고 아무 처리도 안 함.
  외부 트리거(검사 요청)가 오면 inspect_requested를 켜고,
  그 시점 이후 카메라 3장이 새로 들어오는 걸 기다렸다가 한 번만 추론 실행.
  추론이 끝나면 플래그를 다시 끈다.

클래스 구조 (총 20개, 학습된 모델 기준):
  sneaker_{A,B,C}_{left,right}_{240,260,280}  (18개, 종류+좌우+사이즈 통합)
  stain, tear                                  (2개, defect)

  → 사이즈는 별도 계산 없이 클래스명에서 바로 파싱.
  → stain/tear는 어느 신발 소속인지 bbox 겹침으로 판단.

입력:
  /sim/camera/cam1/image_raw, cam2, cam3 (상시 구독, latest frame만 유지)
  /vision/trigger_inspection (검사 요청 트리거, 형태 미정 - TODO)

출력:
  /vision/inspection_result  (ShoeInspectionResult, DB 노드가 구독)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

# TODO: 커스텀 msg 빌드 후 활성화
# from doosan_vision_interfaces.msg import ShoeInspectionResult


# ---------------------------------------------------------------------
# 클래스별 confidence 임계값
# ---------------------------------------------------------------------
# 신발 종류(sneaker_*)는 AP 0.995로 매우 안정적이라 threshold를 높게 잡아도 됨.
# stain(AP 0.58~0.67), tear(AP 0.32~0.35)는 성능이 낮아서 threshold를 낮춰야
# 그나마 recall을 확보함. (학습 데이터 보강이 근본 해결책, 임시 대응)
CONF_THRESHOLD_SHOE = 0.5
CONF_THRESHOLD_DEFECT = 0.2

# predict() 자체에는 두 값 중 더 낮은 값으로 넣고, 이후 클래스별로 재필터링
CONF_THRESHOLD_MIN = min(CONF_THRESHOLD_SHOE, CONF_THRESHOLD_DEFECT)

# defect가 신발 bbox와 겹친다고 판단할 최소 비율
# (defect bbox 면적 중 신발 bbox와 겹치는 비율 = intersection / defect_area)
DEFECT_OVERLAP_THRESHOLD = 0.3


class VisionNode(Node):

    def __init__(self):
        super().__init__('vision_node')

        self.declare_parameter('model_path', '/home/woogi/doosan_pjt3/yolo_models/best.pt')
        self.declare_parameter('image_size', 640)

        self.model_path = self.get_parameter('model_path').value
        self.img_size = self.get_parameter('image_size').value

        self.bridge = CvBridge()
        self.model = self._load_model()

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ---- 최신 프레임 저장용 ----
        self.latest_frames = {'cam1': None, 'cam2': None, 'cam3': None}

        # ---- 검사 요청 플래그 ----
        self.inspect_requested = False

        # ---- 트리거 이후 캡처된 프레임 ----
        self.captured_frames = {'cam1': None, 'cam2': None, 'cam3': None}

        # ---- 카메라 구독 (상시) ----
        self.sub_cam1 = self.create_subscription(
            Image, '/sim/camera/cam1/image_raw',
            lambda msg: self.image_callback(msg, 'cam1'), image_qos,
        )
        self.sub_cam2 = self.create_subscription(
            Image, '/sim/camera/cam2/image_raw',
            lambda msg: self.image_callback(msg, 'cam2'), image_qos,
        )
        self.sub_cam3 = self.create_subscription(
            Image, '/sim/camera/cam3/image_raw',
            lambda msg: self.image_callback(msg, 'cam3'), image_qos,
        )

        # ---- 검사 요청 트리거 구독 ----
        # TODO: 실제 트리거 형태(토픽/서비스) 확정되면 아래 구독 활성화
        # self.sub_trigger = self.create_subscription(
        #     Bool, '/vision/trigger_inspection', self.trigger_callback, 10,
        # )

        # ---- 결과 발행자 ----
        # TODO: 커스텀 msg 타입 생성 후 활성화
        # self.pub_result = self.create_publisher(
        #     ShoeInspectionResult, '/vision/inspection_result', 10,
        # )

        self.get_logger().info(f'Vision node initialized (model: {self.model_path})')

    # ------------------------------------------------------------------
    # 초기화 헬퍼
    # ------------------------------------------------------------------

    def _load_model(self):
        try:
            from ultralytics import YOLO
            model = YOLO(self.model_path)
            self.get_logger().info('YOLO model loaded')
            return model
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {e}')
            return None

    # ------------------------------------------------------------------
    # 트리거 콜백
    # ------------------------------------------------------------------

    def trigger_callback(self, msg=None):
        """검사 요청 수신. 이 시점부터 들어오는 카메라 프레임을 캡처 대상으로 삼는다."""
        if self.inspect_requested:
            self.get_logger().warn('Inspection already in progress, ignoring duplicate trigger')
            return

        self.get_logger().info('Inspection triggered')
        self.captured_frames = {'cam1': None, 'cam2': None, 'cam3': None}
        self.inspect_requested = True

    # ------------------------------------------------------------------
    # 카메라 콜백 (3개)
    # ------------------------------------------------------------------

    def image_callback(self, msg: Image, cam_name: str):
        self.latest_frames[cam_name] = msg

        if not self.inspect_requested:
            return

        self.captured_frames[cam_name] = msg

        if all(v is not None for v in self.captured_frames.values()):
            self._run_inspection(self.captured_frames)
            self.inspect_requested = False
            self.captured_frames = {'cam1': None, 'cam2': None, 'cam3': None}

    # ------------------------------------------------------------------
    # 핵심 로직: 검사 실행
    # ------------------------------------------------------------------

    def _run_inspection(self, frames: dict):
        image_msgs = [frames['cam1'], frames['cam2'], frames['cam3']]

        left_result, right_result = self._infer_pair(image_msgs)
        judgement = self._judge_pair(left_result, right_result)

        self._publish_result(left_result, right_result, judgement)

    # ------------------------------------------------------------------
    # 이미지 3장 → 왼짝/오른짝 종합
    # ------------------------------------------------------------------

    def _infer_pair(self, image_msgs: list):
        """
        카메라 3장을 각각 추론.
        각 프레임 안에서: 신발 detection을 좌/우로 나눠 최고 신뢰도 bbox 선정,
        defect(stain/tear) detection은 그 bbox와의 겹침으로 좌/우 소속 판단.
        최종적으로 3프레임 결과를 종합(클래스는 투표, defect는 OR)한다.
        """
        left_class_votes = {}   # {'A_240': conf합, ...}
        right_class_votes = {}
        left_has_stain = False
        right_has_stain = False
        left_has_tear = False
        right_has_tear = False
        left_best_conf = 0.0
        right_best_conf = 0.0

        for msg in image_msgs:
            if msg is None:
                continue
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception as e:
                self.get_logger().warn(f'CV bridge failed: {e}')
                continue

            dets = self._run_inference(cv_image)

            # 이 프레임 안에서 신발 detection을 좌/우로 분리 (최고 confidence만)
            left_shoe = self._best_shoe_det(dets, side='left')
            right_shoe = self._best_shoe_det(dets, side='right')

            if left_shoe:
                key = f"{left_shoe['color']}_{left_shoe['size']}"
                left_class_votes[key] = left_class_votes.get(key, 0) + left_shoe['confidence']
                left_best_conf = max(left_best_conf, left_shoe['confidence'])

            if right_shoe:
                key = f"{right_shoe['color']}_{right_shoe['size']}"
                right_class_votes[key] = right_class_votes.get(key, 0) + right_shoe['confidence']
                right_best_conf = max(right_best_conf, right_shoe['confidence'])

            # defect(stain/tear)를 좌/우 신발 bbox와 겹침으로 배정
            for d in dets:
                if d['class_name'] not in ('stain', 'tear'):
                    continue
                if d['confidence'] < CONF_THRESHOLD_DEFECT:
                    continue

                side = self._assign_defect_side(d['bbox'], left_shoe, right_shoe)
                if side == 'left':
                    if d['class_name'] == 'stain':
                        left_has_stain = True
                    else:
                        left_has_tear = True
                elif side == 'right':
                    if d['class_name'] == 'stain':
                        right_has_stain = True
                    else:
                        right_has_tear = True
                # side가 None이면 어느 쪽 신발과도 안 겹침 → 배정 보류 (로그만 남김)
                else:
                    self.get_logger().warn(
                        f"Defect '{d['class_name']}' detected but couldn't assign to a shoe side"
                    )

        left_key = max(left_class_votes, key=left_class_votes.get) if left_class_votes else None
        right_key = max(right_class_votes, key=right_class_votes.get) if right_class_votes else None

        left_color, left_size = left_key.split('_') if left_key else (None, 0)
        right_color, right_size = right_key.split('_') if right_key else (None, 0)

        left_result = {
            'color': left_color, 'size': int(left_size) if left_size else 0,
            'has_stain': left_has_stain, 'has_tear': left_has_tear,
            'confidence': left_best_conf,
        }
        right_result = {
            'color': right_color, 'size': int(right_size) if right_size else 0,
            'has_stain': right_has_stain, 'has_tear': right_has_tear,
            'confidence': right_best_conf,
        }
        return left_result, right_result

    # ------------------------------------------------------------------
    # 보조 함수
    # ------------------------------------------------------------------

    @staticmethod
    def _best_shoe_det(dets: list, side: str):
        """
        dets 중 class_name이 'sneaker_{color}_{side}_{size}' 형태인 것들 중
        confidence가 가장 높은 것 하나를 뽑아 파싱해서 반환.
        없으면 None.
        """
        candidates = [
            d for d in dets
            if d['class_name'].startswith('sneaker_') and f'_{side}_' in d['class_name']
        ]
        if not candidates:
            return None

        best = max(candidates, key=lambda d: d['confidence'])
        parts = best['class_name'].split('_')  # ['sneaker', 'A', 'left', '240']
        return {
            'color': parts[1],
            'side': parts[2],
            'size': parts[3],
            'confidence': best['confidence'],
            'bbox': best['bbox'],
        }

    @staticmethod
    def _bbox_intersection_area(box_a, box_b) -> float:
        """두 bbox([x1,y1,x2,y2])의 교집합 면적."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return (ix2 - ix1) * (iy2 - iy1)

    @staticmethod
    def _bbox_area(box) -> float:
        x1, y1, x2, y2 = box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def _assign_defect_side(self, defect_bbox, left_shoe, right_shoe):
        """
        defect bbox가 left_shoe/right_shoe bbox 중 어느 쪽과 더 많이 겹치는지로
        소속을 판단. (intersection 면적 / defect bbox 면적) 비율 기준.
        둘 다 임계값 미달이면 None.
        """
        defect_area = self._bbox_area(defect_bbox)
        if defect_area == 0:
            return None

        left_overlap = 0.0
        right_overlap = 0.0

        if left_shoe:
            left_overlap = self._bbox_intersection_area(defect_bbox, left_shoe['bbox']) / defect_area
        if right_shoe:
            right_overlap = self._bbox_intersection_area(defect_bbox, right_shoe['bbox']) / defect_area

        if left_overlap < DEFECT_OVERLAP_THRESHOLD and right_overlap < DEFECT_OVERLAP_THRESHOLD:
            return None

        return 'left' if left_overlap >= right_overlap else 'right'

    def _run_inference(self, image: np.ndarray) -> list:
        """
        단일 이미지에 YOLO seg 추론.
        낮은 쪽 임계값(CONF_THRESHOLD_MIN)으로 뽑은 뒤, 호출부에서
        클래스별로 다시 필터링해서 쓴다 (신발/defect 임계값이 다르므로).
        """
        if self.model is None:
            return []

        results = self.model.predict(
            image, conf=CONF_THRESHOLD_MIN, imgsz=self.img_size, verbose=False,
        )

        dets = []
        if len(results) > 0:
            r = results[0]
            names = r.names
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                cls_name = names[int(cls)]
                conf_val = float(conf)

                # 클래스 종류에 따라 다른 임계값 적용
                threshold = (
                    CONF_THRESHOLD_DEFECT if cls_name in ('stain', 'tear')
                    else CONF_THRESHOLD_SHOE
                )
                if conf_val < threshold:
                    continue

                dets.append({
                    'class_name': cls_name,
                    'confidence': conf_val,
                    'bbox': box.tolist(),
                })
        return dets

    # ------------------------------------------------------------------
    # 짝 종합 판정
    # ------------------------------------------------------------------

    def _judge_pair(self, left: dict, right: dict) -> dict:
        """
        완전한 정상 짝이 아니면 무조건 폐기.
        """
        if left['has_tear'] or right['has_tear']:
            return {'accepted': False, 'reason': 'defect_tear'}

        if left['has_stain'] or right['has_stain']:
            return {'accepted': False, 'reason': 'defect_stain'}

        if left['color'] is None or right['color'] is None:
            return {'accepted': False, 'reason': 'incomplete_pair'}

        if left['color'] != right['color']:
            return {'accepted': False, 'reason': 'mismatch_class'}

        if left['size'] == 0 or right['size'] == 0:
            return {'accepted': False, 'reason': 'incomplete_pair'}

        if left['size'] != right['size']:
            return {'accepted': False, 'reason': 'mismatch_size'}

        return {'accepted': True, 'reason': 'ok'}

    # ------------------------------------------------------------------
    # 결과 발행
    # ------------------------------------------------------------------

    def _publish_result(self, left: dict, right: dict, judgement: dict):
        # TODO: 커스텀 msg 타입 생성 후 실제 발행으로 교체
        self.get_logger().info(
            f"[RESULT] accepted={judgement['accepted']}, reason={judgement['reason']}, "
            f"L=(color={left['color']}, size={left['size']}, "
            f"stain={left['has_stain']}, tear={left['has_tear']}), "
            f"R=(color={right['color']}, size={right['size']}, "
            f"stain={right['has_stain']}, tear={right['has_tear']})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()