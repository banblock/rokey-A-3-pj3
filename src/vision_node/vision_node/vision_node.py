#!/usr/bin/env python3
"""
Vision 판정 노드.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import numpy as np

from recycle_interfaces.msg import ShoeInspectionResult


CONF_THRESHOLD_SHOE = 0.5
CONF_THRESHOLD_DEFECT = 0.2
CONF_THRESHOLD_MIN = min(CONF_THRESHOLD_SHOE, CONF_THRESHOLD_DEFECT)
DEFECT_OVERLAP_THRESHOLD = 0.3

COLOR_TO_INT = {'red': 0, 'green': 1, 'yellow': 2}


class VisionNode(Node):

    def __init__(self):
        super().__init__('vision_node')

        self.declare_parameter('model_path', '/home/rokey/cobot3_ws/src/vision_node/resource/best.pt')
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

        self.latest_frames = {'cam1': None, 'cam2': None, 'cam3': None}
        self.inspect_requested = False
        self.captured_frames = {'cam1': None, 'cam2': None, 'cam3': None}

        # ---- 카메라 구독 (상시) ----
        self.sub_cam1 = self.create_subscription(
            Image, '/d455_1/color/image_raw',
            lambda msg: self.image_callback(msg, 'cam1'), image_qos,
        )
        self.sub_cam2 = self.create_subscription(
            Image, '/d455_2/color/image_raw',
            lambda msg: self.image_callback(msg, 'cam2'), image_qos,
        )
        self.sub_cam3 = self.create_subscription(
            Image, '/d455_3/color/image_raw',
            lambda msg: self.image_callback(msg, 'cam3'), image_qos,
        )

        # ---- 검사 요청 서비스 서버 ----
        # self.srv_trigger = self.create_service(
        #     InspectShoePair,
        #     '/vision/inspect_shoe_pair',
        #     self.trigger_callback,
        # )

        self.sub_trigger = self.create_subscription(
            Bool, '/shoe_stop',
            lambda msg: self.trigger_callback(msg), 10,
        )

        # ---- 결과 발행자 (DB 노드로) ----
        self.pub_result = self.create_publisher(
            ShoeInspectionResult, '/vision/inspection_result', 10,
        )

        self.get_logger().info(f'Vision node initialized (model: {self.model_path})')

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
    # 트리거 콜백 (서비스)
    # ------------------------------------------------------------------

    def trigger_callback(self, msg):
        self.get_logger().info(f'{msg}')
        """검사 요청 수신 (서비스). 이 시점부터 들어오는 카메라 프레임을 캡처 대상으로 삼는다."""
        if msg.data is False:
            self.get_logger().info('Inspection request canceled')
            self.inspect_requested = False
            self.captured_frames = {'cam1': None, 'cam2': None, 'cam3': None}
            return
        self.get_logger().info('Inspection triggered')
        self.captured_frames = {'cam1': None, 'cam2': None, 'cam3': None}
        self.inspect_requested = True

        return

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
    # def _run_inspection(self, frames: dict):
    #     image_msgs = [frames['cam1'], frames['cam2'], frames['cam3']]
    #     left_result, right_result = self._infer_pair(image_msgs)
    #     judgement = self._judge_pair(left_result, right_result)
    #     self._publish_result(left_result, right_result, judgement)

    def _run_inspection(self, frames: dict):
        image_msgs = [frames['cam1'], frames['cam2']]
        # frames['cam3']]
        result = self._infer_pair(image_msgs)
        judgement = self._judge_pair(result)
        self._publish_result(result, judgement)

    # ------------------------------------------------------------------
    # def _infer_pair(self, image_msgs: list):
    #     left_class_votes = {}
    #     right_class_votes = {}
    #     left_has_stain = False
    #     right_has_stain = False
    #     left_has_tear = False
    #     right_has_tear = False
    #     left_best_conf = 0.0
    #     right_best_conf = 0.0

    #     for msg in image_msgs:
    #         if msg is None:
    #             continue
    #         try:
    #             cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    #         except Exception as e:
    #             self.get_logger().warn(f'CV bridge failed: {e}')
    #             continue

    #         dets = self._run_inference(cv_image)

    #         left_shoe = self._best_shoe_det(dets, side='left')
    #         right_shoe = self._best_shoe_det(dets, side='right')

    #         self.get_logger().info(
    #             f"[DEBUG] left_shoe={left_shoe}, right_shoe={right_shoe}, "
    #             f"total_dets={len(dets)}, det_classes={[d['class_name'] for d in dets]}"
    #         )

    #         if left_shoe:
    #             key = f"{left_shoe['color']}_{left_shoe['size']}"
    #             left_class_votes[key] = left_class_votes.get(key, 0) + left_shoe['confidence']
    #             left_best_conf = max(left_best_conf, left_shoe['confidence'])

    #         if right_shoe:
    #             key = f"{right_shoe['color']}_{right_shoe['size']}"
    #             right_class_votes[key] = right_class_votes.get(key, 0) + right_shoe['confidence']
    #             right_best_conf = max(right_best_conf, right_shoe['confidence'])

    #         for d in dets:
    #             if d['class_name'] not in ('stain', 'tear'):
    #                 continue
    #             if d['confidence'] < CONF_THRESHOLD_DEFECT:
    #                 continue

    #             side = self._assign_defect_side(d['bbox'], left_shoe, right_shoe)
    #             if side == 'left':
    #                 if d['class_name'] == 'stain':
    #                     left_has_stain = True
    #                 else:
    #                     left_has_tear = True
    #             elif side == 'right':
    #                 if d['class_name'] == 'stain':
    #                     right_has_stain = True
    #                 else:
    #                     right_has_tear = True
    #             else:
    #                 self.get_logger().warn(
    #                     f"Defect '{d['class_name']}' detected but couldn't assign to a shoe side"
    #                 )

    #     left_key = max(left_class_votes, key=left_class_votes.get) if left_class_votes else None
    #     right_key = max(right_class_votes, key=right_class_votes.get) if right_class_votes else None

    #     left_color, left_size = left_key.split('_') if left_key else (None, 0)
    #     right_color, right_size = right_key.split('_') if right_key else (None, 0)

    #     left_result = {
    #         'color': left_color, 'size': int(left_size) if left_size else 0,
    #         'has_stain': left_has_stain, 'has_tear': left_has_tear,
    #         'confidence': left_best_conf,
    #     }
    #     right_result = {
    #         'color': right_color, 'size': int(right_size) if right_size else 0,
    #         'has_stain': right_has_stain, 'has_tear': right_has_tear,
    #         'confidence': right_best_conf,
    #     }
    #     return left_result, right_result

    def _infer_pair(self, image_msgs: list):
    # """켤레(sneakers_0) 하나로 탐지, tear는 그 박스와 겹치는지로 판단."""
        shoe_found = False
        shoe_best_conf = 0.0
        has_tear = False

        for msg in image_msgs:
            if msg is None:
                continue
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception as e:
                self.get_logger().warn(f'CV bridge failed: {e}')
                continue

            dets = self._run_inference(cv_image)

            shoe_pair = self._best_shoe_pair_det(dets)

            self.get_logger().info(
                f"[DEBUG] shoe_pair={shoe_pair}, "
                f"total_dets={len(dets)}, det_classes={[d['class_name'] for d in dets]}"
            )

            if shoe_pair:
                shoe_found = True
                shoe_best_conf = max(shoe_best_conf, shoe_pair['confidence'])

            for d in dets:
                if d['class_name'] != 'tear':
                    continue
                if d['confidence'] < CONF_THRESHOLD_DEFECT:
                    continue

                if shoe_pair is not None:
                    overlap = self._bbox_intersection_area(d['bbox'], shoe_pair['bbox']) / self._bbox_area(d['bbox'])
                    if overlap >= DEFECT_OVERLAP_THRESHOLD:
                        has_tear = True
                    else:
                        self.get_logger().warn("tear detected but doesn't overlap with shoe pair bbox")
                else:
                    self.get_logger().warn("tear detected but no shoe pair found in this frame")

        result = {
            'shoe_found': shoe_found,
            'has_tear': has_tear,
            'confidence': shoe_best_conf,
        }
        return result

    # ------------------------------------------------------------------
    @staticmethod
    # def _best_shoe_det(dets: list, side: str):
    #     candidates = [
    #         d for d in dets
    #         if d['class_name'].startswith('sneaker_') and f'_{side}_' in d['class_name']
    #     ]
    #     if not candidates:
    #         return None

    #     best = max(candidates, key=lambda d: d['confidence'])
    #     parts = best['class_name'].split('_')
    #     return {
    #         'color': parts[1],
    #         'side': parts[2],
    #         'size': parts[3],
    #         'confidence': best['confidence'],
    #         'bbox': best['bbox'],
    #     }

    @staticmethod
    def _best_shoe_pair_det(dets: list):
        """sneakers_0(켤레 전체) 중 confidence 가장 높은 것 하나."""
        candidates = [d for d in dets if d['class_name'] == 'sneakers_0']
        if not candidates:
            return None

        best = max(candidates, key=lambda d: d['confidence'])
        return {
            'confidence': best['confidence'],
            'bbox': best['bbox'],
        }

    @staticmethod
    def _bbox_intersection_area(box_a, box_b) -> float:
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

    # def _assign_defect_side(self, defect_bbox, left_shoe, right_shoe):
    #     defect_area = self._bbox_area(defect_bbox)
    #     if defect_area == 0:
    #         return None

    #     left_overlap = 0.0
    #     right_overlap = 0.0

    #     if left_shoe:
    #         left_overlap = self._bbox_intersection_area(defect_bbox, left_shoe['bbox']) / defect_area
    #     if right_shoe:
    #         right_overlap = self._bbox_intersection_area(defect_bbox, right_shoe['bbox']) / defect_area

    #     if left_overlap < DEFECT_OVERLAP_THRESHOLD and right_overlap < DEFECT_OVERLAP_THRESHOLD:
    #         return None

    #     return 'left' if left_overlap >= right_overlap else 'right'

    def _run_inference(self, image: np.ndarray) -> list:
        if self.model is None:
            return []

        results = self.model.predict(
            image, conf=CONF_THRESHOLD_MIN, imgsz=self.img_size,
            iou=0.5,  # NMS IoU 임계값 명시
            verbose=False,
        )

        import cv2
        import time
        annotated = results[0].plot()
        debug_path = f'/home/rokey/debug_inference/debug_inference_{time.time()}.png'
        cv2.imwrite(debug_path, annotated)

        dets = []
        if len(results) > 0:
            r = results[0]
            names = r.names
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                cls_name = names[int(cls)]
                conf_val = float(conf)

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
    # def _judge_pair(self, left: dict, right: dict) -> dict:
    #     if left['has_tear'] or right['has_tear']:
    #         self.get_logger().info('Discard reason: defect_tear')
    #         return {'discard': True}

    #     if left['has_stain'] or right['has_stain']:
    #         self.get_logger().info('Discard reason: defect_stain')
    #         return {'discard': True}

    #     if left['color'] is None or right['color'] is None:
    #         self.get_logger().info('Discard reason: incomplete_pair')
    #         return {'discard': True}

    #     if left['color'] != right['color']:
    #         self.get_logger().info('Discard reason: mismatch_class')
    #         return {'discard': True}

    #     if left['size'] == 0 or right['size'] == 0:
    #         self.get_logger().info('Discard reason: incomplete_pair (size)')
    #         return {'discard': True}

    #     if left['size'] != right['size']:
    #         self.get_logger().info('Discard reason: mismatch_size')
    #         return {'discard': True}

    #     return {'discard': False}

    # def _judge_pair(self, left: dict, right: dict) -> dict:
    #     discard = False

    #     if left['has_tear'] or right['has_tear']:
    #         discard = True
    #     elif left['has_stain'] or right['has_stain']:
    #         discard = True
    #     elif left['color'] is None or right['color'] is None:
    #         discard = True
    #     elif left['color'] != right['color']:
    #         discard = True
    #     elif left['size'] == 0 or right['size'] == 0:
    #         discard = True
    #     elif left['size'] != right['size']:
    #         discard = True

    #     color_str = left['color'] or right['color']
    #     color_int = COLOR_TO_INT.get(color_str, -1)
    #     size = left['size'] if left['size'] else right['size']

    #     return {'discard': discard, 'color': color_int, 'size': size}

    def _judge_pair(self, result: dict) -> dict:
        """
        TODO: 색상 일치 판정(OpenCV 픽셀 카운팅)은 추후 추가.
        지금은 신발 감지 여부 + tear 유무만으로 판정.
        """
        if not result['shoe_found']:
            return {'discard': True, 'reason': 'no_shoe_detected'}

        if result['has_tear']:
            return {'discard': True, 'reason': 'defect_tear'}

        return {'discard': False, 'reason': 'ok'}

    # ------------------------------------------------------------------
    # 결과 발행
    # ------------------------------------------------------------------

    # def _publish_result(self, left: dict, right: dict, judgement: dict):
    #     msg = ShoeInspectionResult()
    #     msg.discard = judgement['discard']
    #     msg.left_color = left['color'] or ''
    #     msg.left_size = left['size']
    #     msg.right_color = right['color'] or ''
    #     msg.right_size = right['size']
    #     self.pub_result.publish(msg)

    #     self.get_logger().info(
    #         f"[RESULT] discard={judgement['discard']}, "
    #         f"L=(color={left['color']}, size={left['size']}), "
    #         f"R=(color={right['color']}, size={right['size']})"
    #     )

    # def _publish_result(self, left: dict, right: dict, judgement: dict):
    #     msg = ShoeInspectionResult()
    #     msg.discard = judgement['discard']
    #     msg.color = judgement['color']
    #     msg.size = judgement['size']
    #     self.pub_result.publish(msg)

    #     self.get_logger().info(
    #         f"[RESULT] discard={judgement['discard']}, "
    #         f"color={judgement['color']}, size={judgement['size']}"
    #     )

    def _publish_result(self, result: dict, judgement: dict):
        msg = ShoeInspectionResult()
        msg.discard = judgement['discard']
        msg.color = 0  # TODO: OpenCV 색상 판정 추가 후 채울 것
        msg.size = 240    # TODO: cam3 seg 기반 사이즈 측정 추가 후 채울 것
        self.pub_result.publish(msg)

        self.get_logger().info(
            f"[RESULT] discard={judgement['discard']}, reason={judgement['reason']}"
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