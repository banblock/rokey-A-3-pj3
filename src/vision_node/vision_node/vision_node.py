#!/usr/bin/env python3
"""
Vision 판정 노드.
"""

import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import numpy as np
import cv2
import time

from recycle_interfaces.msg import ShoeInspectionResult


CONF_THRESHOLD_SHOE = 0.5
CONF_THRESHOLD_DEFECT = 0.2
CONF_THRESHOLD_MIN = min(CONF_THRESHOLD_SHOE, CONF_THRESHOLD_DEFECT)
DEFECT_OVERLAP_THRESHOLD = 0.3

#COLOR_TO_INT = {'red': 0, 'green': 1, 'yellow': 2}


class VisionNode(Node):

    def __init__(self):
        super().__init__('vision_node')

        # 1단계: shoe 검출 (원본 이미지 전체). 2단계: 1단계가 찾은 shoe 박스마다 크롭+확대해서
        # tear 전용 모델로 재검출 (원본에서 tear가 너무 작아 1단계만으론 놓치는 문제 보완).
        self.declare_parameter('model_path', '/home/rokey/cobot3_ws/src/vision_node/resource/best_task1.pt')
        self.declare_parameter('model_path_stage2', '/home/rokey/cobot3_ws/src/vision_node/resource/best_task2.pt')
        self.declare_parameter('image_size', 960)
        self.declare_parameter('image_size_stage2', 480)
        self.declare_parameter('stage2_pad_ratio', 0.3)
        # 검사할 때마다 디스크에 디버그 이미지를 저장하면(cv2.imwrite) I/O 때문에 느려진다.
        # publish는 가벼우니 기본으로 두고, 디스크 저장은 필요할 때만 켠다.
        self.declare_parameter('save_debug_image', True)

        # 3단계: cam3(바닥) OBB 모델로 신발 길이(px) 측정 -> 사이즈 분류.
        # px->mm 캘리브레이션이 아직 없어서, 일단 px 길이 기준 임의 구간 2개로
        # 240/260/280mm 3개를 나눈다 (실측 캘리브레이션 끝나면 이 두 임계값만 바꾸면 됨).
        self.declare_parameter('model_path_stage3', '/home/rokey/cobot3_ws/src/vision_node/resource/best_task3.pt')
        self.declare_parameter('image_size_stage3', 960)
        self.declare_parameter('size_px_threshold_1', 300.0)  # 이 미만 -> 240mm
        self.declare_parameter('size_px_threshold_2', 340.0)  # 이상 -> 280mm, 사이는 260mm

        self.model_path = self.get_parameter('model_path').value
        self.model_path_stage2 = self.get_parameter('model_path_stage2').value
        self.img_size = self.get_parameter('image_size').value
        self.img_size_stage2 = self.get_parameter('image_size_stage2').value
        self.stage2_pad_ratio = self.get_parameter('stage2_pad_ratio').value
        self.save_debug_image = self.get_parameter('save_debug_image').value
        self.model_path_stage3 = self.get_parameter('model_path_stage3').value
        self.img_size_stage3 = self.get_parameter('image_size_stage3').value
        self.size_px_threshold_1 = self.get_parameter('size_px_threshold_1').value
        self.size_px_threshold_2 = self.get_parameter('size_px_threshold_2').value

        self.bridge = CvBridge()
        self.model = self._load_model(self.model_path, 'stage1 (shoe)')
        self.model2 = self._load_model(self.model_path_stage2, 'stage2 (tear crop)')
        self.model3 = self._load_model(self.model_path_stage3, 'stage3 (bottom OBB size)')
        self._warmup_models()

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.latest_frames = {'cam1': None, 'cam2': None, 'cam3': None}
        self.inspect_requested = False
        self.captured_frames = {'cam1': None, 'cam2': None, 'cam3': None}
        # image_callback -> _run_inspection(YOLO 추론)이 몇 초씩 걸리는 동안, 같은
        # 콜백그룹에 있으면 /shoe_stop 취소 요청이 그 뒤에 밀려서 처리된다. 트리거를
        # 별도의(상호배제) 콜백그룹으로 분리하고 MultiThreadedExecutor로 돌려야
        # 추론 도중에도 취소 요청이 바로 처리된다. 두 그룹이 동시에 돌 수 있으니
        # 공유 상태(captured_frames/inspect_requested)는 락으로 보호한다.
        self.state_lock = threading.Lock()
        camera_cb_group = MutuallyExclusiveCallbackGroup()
        trigger_cb_group = MutuallyExclusiveCallbackGroup()

        # ---- 카메라 구독 (상시) ----
        self.sub_cam1 = self.create_subscription(
            Image, '/d455_1/color/image_raw',
            lambda msg: self.image_callback(msg, 'cam1'), image_qos,
            callback_group=camera_cb_group,
        )
        self.sub_cam2 = self.create_subscription(
            Image, '/d455_2/color/image_raw',
            lambda msg: self.image_callback(msg, 'cam2'), image_qos,
            callback_group=camera_cb_group,
        )
        self.sub_cam3 = self.create_subscription(
            Image, '/d455_3/color/image_raw',
            lambda msg: self.image_callback(msg, 'cam3'), image_qos,
            callback_group=camera_cb_group,
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
            callback_group=trigger_cb_group,
        )

        # ---- 결과 발행자 (DB 노드로) ----
        self.pub_result = self.create_publisher(
            ShoeInspectionResult, '/vision/inspection_result', 10,
        )

        self.get_logger().info(f'Vision node initialized (model: {self.model_path})')

        self.pub_result_img1 = self.create_publisher(Image, '/vision/result_img1', 10)
        self.pub_result_img2 = self.create_publisher(Image, '/vision/result_img2', 10)
        self.pub_result_img3 = self.create_publisher(Image, '/vision/result_img3', 10)

    # ------------------------------------------------------------------
    def _load_model(self, path, label):
        try:
            from ultralytics import YOLO
            model = YOLO(path)
            self.get_logger().info(f'YOLO model loaded [{label}]: {path}')
            return model
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model [{label}] ({path}): {e}')
            return None

    def _warmup_models(self):
        """YOLO는 로드 직후 첫 predict 호출에서 CUDA 커널/cuDNN 알고리즘을 고르느라 몇 초씩
        더 걸린다. 노드 초기화 시점에 더미 이미지로 한 번 미리 호출해둬서, 실제 검사 요청이
        왔을 때는 이 워밍업 비용을 안 치르게 한다."""
        dummy1 = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        dummy2 = np.zeros((self.img_size_stage2, self.img_size_stage2, 3), dtype=np.uint8)
        dummy3 = np.zeros((self.img_size_stage3, self.img_size_stage3, 3), dtype=np.uint8)
        if self.model is not None:
            self.model.predict(dummy1, imgsz=self.img_size, quantize="fp16", verbose=False)
        if self.model2 is not None:
            self.model2.predict(dummy2, imgsz=self.img_size_stage2, quantize="fp16", verbose=False)
        if self.model3 is not None:
            self.model3.predict(dummy3, imgsz=self.img_size_stage3, quantize="fp16", verbose=False)
        self.get_logger().info('Model warm-up done')

    # ------------------------------------------------------------------
    # 트리거 콜백 (서비스)
    # ------------------------------------------------------------------

    def trigger_callback(self, msg):
        self.get_logger().info(f'{msg}')
        """검사 요청 수신 (서비스). 이 시점부터 들어오는 카메라 프레임을 캡처 대상으로 삼는다."""
        with self.state_lock:
            if msg.data is False:
                self.get_logger().info('Inspection request canceled')
                self.inspect_requested = False
                self.captured_frames = {'cam1': None, 'cam2': None, 'cam3': None}
                return
            self.get_logger().info('Inspection triggered')
            self.captured_frames = {'cam1': None, 'cam2': None, 'cam3': None}
            self.inspect_requested = True

    # ------------------------------------------------------------------
    # 카메라 콜백 (3개)
    # ------------------------------------------------------------------

    def image_callback(self, msg: Image, cam_name: str):
        self.latest_frames[cam_name] = msg

        # 상태 확인/갱신만 락으로 짧게 보호하고, 실제 추론(_run_inspection, 몇 초 걸림)은
        # 락 밖에서 돌려야 그 사이에 /shoe_stop 취소 콜백(다른 스레드)이 안 막힌다.
        frames_to_process = None
        with self.state_lock:
            if not self.inspect_requested:
                return
            self.captured_frames[cam_name] = msg
            # _run_inspection은 cam1/cam2만 쓴다 (cam3는 사이즈 측정용으로 나중에 쓸 예정,
            # TODO 상태 — 지금 여기 넣으면 cam3 토픽이 안 들어올 때 검사가 영영 안 끝난다).
            if all(self.captured_frames[c] is not None for c in ('cam1', 'cam2')):
                frames_to_process = self.captured_frames
                self.inspect_requested = False
                self.captured_frames = {'cam1': None, 'cam2': None, 'cam3': None}

        if frames_to_process is not None:
            self._run_inspection(frames_to_process)

    # ------------------------------------------------------------------
    # def _run_inspection(self, frames: dict):
    #     image_msgs = [frames['cam1'], frames['cam2'], frames['cam3']]
    #     left_result, right_result = self._infer_pair(image_msgs)
    #     judgement = self._judge_pair(left_result, right_result)
    #     self._publish_result(left_result, right_result, judgement)

    def _run_inspection(self, frames: dict):
        start = time.time()
        self.get_logger().info('Inference started')
        try:
            image_msgs = [frames['cam1'], frames['cam2']]
            result = self._infer_pair(image_msgs)
            judgement = self._judge_pair(result)
            # cam3(바닥)는 cam1/cam2처럼 검사 완료 조건에 안 걸려있다 — 안 들어와도
            # 검사 자체가 멈추면 안 되니, 이 시점 기준 가장 최근 프레임을 best-effort로
            # 쓴다 (frames에 캡처된 게 있으면 그걸, 없으면 latest_frames로 대체).
            cam3_msg = frames.get('cam3') or self.latest_frames.get('cam3')
            size_mm = self._measure_shoe_size_mm(cam3_msg)
            self._publish_result(result, judgement, size_mm)
        finally:
            elapsed = time.time() - start
            self.get_logger().info(f'Inference finished ({elapsed:.2f}s)')

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
        # """켤레(shoe) 하나로 탐지, tear는 그 박스와 겹치는지로 판단."""
        shoe_found = False
        shoe_best_conf = 0.0
        has_tear = False

        all_detections = []  #모든 프레임의 탐지 결과를 모을 빈 리스트 생성

        cv_images = []
        for msg in image_msgs:
            if msg is None:
                continue
            try:
                cv_images.append(self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'))
            except Exception as e:
                self.get_logger().warn(f'CV bridge failed: {e}')

        # cam1/cam2를 따로따로 순차 predict하지 않고 한 번에 배치로 돌린다 (1단계도,
        # 크롭해서 돌리는 2단계도) — GPU를 훨씬 효율적으로 써서 검사 1회당 걸리는
        # 시간이 크게 줄어든다.
        dets_per_image = self._run_batched_inference(cv_images)

        for dets in dets_per_image:
            all_detections.extend(dets)  #현재 프레임의 탐지 결과를 전체 리스트에 누적

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
            'detections': all_detections  #_publish_result로 전달하기 위해 result에 추가
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
        """shoe(켤레 전체 — SDG에서 신발 두 짝을 한 프림으로 묶어 labeling함) 중
        confidence 가장 높은 것 하나. 실제 배치 환경에선 켤레 하나만 지나가므로
        한 프레임에 shoe가 여러 개 나올 상황은 거의 없다."""
        candidates = [d for d in dets if d['class_name'] == 'shoe']
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

    def _crop_with_padding(self, image: np.ndarray, bbox, pad_ratio: float):
        h, w = image.shape[:2]
        x0, y0, x1, y1 = bbox
        bw, bh = x1 - x0, y1 - y0
        pad_x, pad_y = bw * pad_ratio, bh * pad_ratio
        cx0 = max(0, int(x0 - pad_x))
        cy0 = max(0, int(y0 - pad_y))
        cx1 = min(w, int(x1 + pad_x))
        cy1 = min(h, int(y1 + pad_y))
        return image[cy0:cy1, cx0:cx1], (cx0, cy0)

    def _run_batched_inference(self, images: list) -> list:
        """1단계(shoe, 원본 전체) -> 검출된 shoe 박스마다 크롭+확대 -> 2단계(tear 전용) 순서로
        돈다. tear가 원본 해상도에서 너무 작아 1단계만으론 컨피던스가 낮거나 아예 후보가
        안 나오는 문제를, shoe는 항상 잘 잡힌다는 전제로 shoe 박스 기준 크롭으로 보완한다
        (infer_cascade.py와 동일한 구조).

        cam1/cam2 이미지를 한 장씩 순차로 predict하면 GPU를 놀리는 시간이 많아서, 1단계는
        여러 이미지를 한 번에 배치로, 2단계도 이미지들에서 나온 crop을 전부 모아 한 번에
        배치로 돌린다. 반환값은 입력 images와 같은 순서의 dets 리스트."""
        n = len(images)
        if self.model is None or n == 0:
            return [[] for _ in range(n)]

        stage1_results = self.model.predict(
            images, conf=CONF_THRESHOLD_SHOE, imgsz=self.img_size,
            iou=0.5, quantize="fp16", verbose=False,
        )

        dets_per_image = [[] for _ in range(n)]
        crops, crop_meta = [], []  # crop_meta: (image_idx, offset_x, offset_y)

        for img_idx, (image, res) in enumerate(zip(images, stage1_results)):
            names1 = res.names
            for box, cls, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
                cls_name = names1[int(cls)]
                conf_val = float(conf)
                if conf_val < CONF_THRESHOLD_SHOE:
                    continue
                bbox = box.tolist()
                dets_per_image[img_idx].append({'class_name': cls_name, 'confidence': conf_val, 'bbox': bbox})
                if cls_name == 'shoe':
                    crop, (ox, oy) = self._crop_with_padding(image, bbox, self.stage2_pad_ratio)
                    if crop.size > 0:
                        crops.append(crop)
                        crop_meta.append((img_idx, ox, oy))

        if self.model2 is not None and crops:
            stage2_results = self.model2.predict(
                crops, conf=CONF_THRESHOLD_DEFECT, imgsz=self.img_size_stage2,
                iou=0.5, quantize="fp16", verbose=False,
            )
            for (img_idx, ox, oy), res in zip(crop_meta, stage2_results):
                names2 = res.names
                for box, cls, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
                    cls_name = names2[int(cls)]
                    conf_val = float(conf)
                    if conf_val < CONF_THRESHOLD_DEFECT:
                        continue
                    cx0, cy0, cx1, cy1 = box.tolist()
                    dets_per_image[img_idx].append({
                        'class_name': cls_name,
                        'confidence': conf_val,
                        'bbox': [cx0 + ox, cy0 + oy, cx1 + ox, cy1 + oy],
                    })

        for image, dets in zip(images, dets_per_image):
            annotated = self._draw_dets(image, dets)
            img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            self.pub_result_img1.publish(img_msg)
            if self.save_debug_image:
                debug_path = f'/home/rokey/debug_inference/debug_inference_{time.time()}.png'
                cv2.imwrite(debug_path, annotated)

        return dets_per_image

    @staticmethod
    def _draw_dets(image: np.ndarray, dets: list) -> np.ndarray:
        out = image.copy()
        colors = {'shoe': (0, 255, 0), 'tear': (0, 0, 255)}
        for d in dets:
            x0, y0, x1, y1 = [int(v) for v in d['bbox']]
            color = colors.get(d['class_name'], (0, 255, 255))
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
            cv2.putText(out, f"{d['class_name']} {d['confidence']:.2f}", (x0, max(0, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return out

    # ------------------------------------------------------------------
    # 3단계: cam3(바닥) OBB로 신발 길이 측정 -> 사이즈 분류
    # ------------------------------------------------------------------

    def _classify_size_mm(self, length_px: float) -> int:
        """px 길이를 240/260/280mm 중 하나로 분류한다. 임계값 두 개(size_px_threshold_1/2)는
        아직 캘리브레이션 전이라 임의值 — 실측 후 이 두 파라미터만 바꾸면 된다."""
        if length_px < self.size_px_threshold_1:
            return 240
        elif length_px < self.size_px_threshold_2:
            return 260
        else:
            return 280

    def _measure_shoe_size_mm(self, cam3_msg):
        """cam3 이미지에서 OBB 모델(model3)로 신발을 찾아 긴 변(px)을 재고, 그걸 사이즈
        구간으로 분류한다. cam3 프레임이 없거나 신발을 못 찾으면 None을 반환한다."""
        if self.model3 is None or cam3_msg is None:
            return None
        try:
            cv_image = self.bridge.imgmsg_to_cv2(cam3_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'CV bridge failed (cam3): {e}')
            return None

        result = self.model3.predict(
            cv_image, imgsz=self.img_size_stage3, quantize="fp16", verbose=False,
        )[0]
        obb = result.obb
        if obb is None or len(obb) == 0:
            self.get_logger().warn('cam3: shoe OBB not found')
            return None

        # confidence 가장 높은 것 하나만 쓴다 (한 번에 켤레 하나만 지나간다는 전제).
        best_idx = int(obb.conf.argmax())
        w, h = float(obb.xywhr[best_idx][2]), float(obb.xywhr[best_idx][3])
        length_px = max(w, h)
        size_mm = self._classify_size_mm(length_px)
        self.get_logger().info(f'cam3: length_px={length_px:.1f} -> size={size_mm}mm')
        return size_mm

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

    def _publish_result(self, result: dict, judgement: dict, size_mm=None):
        msg = ShoeInspectionResult()
        msg.discard = judgement['discard']
        msg.color = 0  # TODO: OpenCV 색상 판정
        # cam3 OBB로 못 재면(카메라 안 들어옴/신발 못 찾음) 240mm 기본값으로 둔다.
        msg.size = size_mm if size_mm is not None else 240

        # ----------------------------------------------------
        # _infer_pair에서 넘어온 result 딕셔너리에서 탐지 결과 추출
        # ----------------------------------------------------
        dets = result.get('detections', [])

        self.pub_result.publish(msg)

        self.get_logger().info(
            f"[RESULT] discard={judgement['discard']}, reason={judgement['reason']}, "
            f"found_defects={len(dets)}, size_mm={msg.size}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    # 카메라 콜백그룹과 트리거 콜백그룹이 서로 다른 스레드에서 동시에 돌 수 있어야
    # 추론(_run_inspection) 중에도 /shoe_stop 취소가 바로 처리된다 — 최소 스레드 2개.
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()