import os
import sys
import glob
import math
import time
import signal
import logging
import argparse
import requests
import xml.etree.ElementTree as ET
import cv2
import torch
import psutil  # 자원 모니터링을 위해 추가
from logging.handlers import RotatingFileHandler
from unittest.mock import MagicMock, patch  # 단위 테스트를 위해 추가

# YOLOv5 레포지토리 함수 import (사전 `pip install -e .` 필요)
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords


def setup_logging():
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
    root_logger = logging.getLogger()
    # 내가 등록하려는 핸들러 종류가 없을 때만 설정
    if not any(
            isinstance(h, (logging.handlers.RotatingFileHandler, logging.StreamHandler)) for h in root_logger.handlers):
        root_logger.setLevel(logging.INFO)
        file_handler = RotatingFileHandler('tracker_session.log', maxBytes=10 ** 7, backupCount=5)
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        root_logger.addHandler(console_handler)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


class VehicleTracker:
    DEFAULT_CLASSES = [2, 3, 5, 7]

    def __init__(self, args):
        setup_logging()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._shutdown_request = False
        signal.signal(signal.SIGINT, self._graceful_shutdown)

        self.model = self._load_model(self.args.weights)
        self._validate_class_ids()
        self.all_cams = self._fetch_cctv_list()

        self.current_cam = None;
        self.cap = None;
        self.tracker = None
        self.is_tracking = False;
        self.boxes_to_draw = [];
        self.frame_id = 0
        self.current_frame = None;
        self.last_fps_time = time.time();
        self.fps = 0
        self.process = psutil.Process(os.getpid())  # 현재 프로세스 정보

    def _graceful_shutdown(self, signum, frame):
        logging.warning("종료 신호(Ctrl+C) 수신. 안전하게 종료를 시도합니다...")
        self._shutdown_request = True

    def _load_model(self, weights_path):
        # ... (이전과 동일) ...
        if not weights_path or not os.path.exists(weights_path):
            logging.error(f"가중치 파일을 찾을 수 없습니다: {weights_path}")
            return None
        try:
            model = attempt_load(weights_path, map_location=self.device)
            if not hasattr(model, 'stride') or not hasattr(model, 'names'):
                raise AttributeError("로드된 파일이 유효한 YOLOv5 모델 객체가 아닙니다.")
            model.to(self.device).eval()
            logging.info(f"모델 로드 완료: {weights_path}")
            return model
        except Exception as e:
            logging.critical(f"모델 로드 중 심각한 오류 발생: {e}")
            return None

    def _validate_class_ids(self):
        # ... (이전과 동일) ...
        if not self.model: return
        valid_ids = []
        max_id = len(self.model.names) - 1
        for cid in self.args.classes:
            if 0 <= cid <= max_id:
                valid_ids.append(cid)
            else:
                logging.warning(f"잘못된 클래스 ID '{cid}'는 무시됩니다. (유효 범위: 0-{max_id})")

        if not valid_ids:
            logging.error("유효한 클래스 ID가 없습니다. 기본값으로 복원합니다.")
            valid_ids = self.DEFAULT_CLASSES

        self.args.classes = valid_ids
        logging.info(f"탐지할 최종 클래스: {[self.model.names[i] for i in self.args.classes]}")

    def _fetch_cctv_list(self):
        # ... (이전과 동일) ...
        cams = [];
        page = 1
        logging.info("ITS API에서 CCTV 목록을 가져오는 중...")
        while page <= self.args.max_pages:
            url = f"http://openapi.its.go.kr:8081/api/NCCTVInfo?key={self.args.api_key}&ReqType=2&type=ex&pageNo={page}"
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                root = ET.fromstring(r.content)
                items = root.findall('data')
                if not items: break
                for it in items:
                    u_node = it.find('cctvurl')
                    if u_node is not None and u_node.text:
                        cams.append({'id': it.find('cctvname').text, 'lat': float(it.find('coordy').text),
                                     'lon': float(it.find('coordx').text), 'url': u_node.text})
                page += 1
            except requests.exceptions.RequestException as e:
                logging.error(f"CCTV API 요청 오류 (Page {page}): {e}");
                break
            except ET.ParseError as e:
                logging.error(f"CCTV API 응답 파싱 오류 (Page {page}): {e}");
                break
        if page > self.args.max_pages: logging.warning(f"최대 페이지({self.args.max_pages})에 도달했습니다.")
        return cams

    def _on_mouse(self, event, x, y, flags, param):
        # ... (이전과 동일) ...
        if event == cv2.EVENT_LBUTTONDOWN and not self.is_tracking:
            for (x1, y1, x2, y2) in self.boxes_to_draw:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.tracker = cv2.TrackerCSRT_create()
                    self.tracker.init(self.current_frame, (x1, y1, x2 - x1, y2 - y1))
                    self.is_tracking = True
                    logging.info(f"추적 시작: Box({x1},{y1},{x2},{y2})")
                    break

    def _choose_next_cctv(self):
        # ... (이전과 동일) ...
        candidates = []
        if not self.current_cam: return None
        for c in self.all_cams:
            if c['id'] == self.current_cam['id']: continue
            d = haversine(self.current_cam['lat'], self.current_cam['lon'], c['lat'], c['lon'])
            if d <= self.args.radius:
                candidates.append((d, c))
        if not candidates: return None
        return sorted(candidates, key=lambda x: x[0])[0]

    def _switch_to_next_cam(self):
        # ... (이전과 동일) ...
        logging.info("다음 CCTV로 전환을 시도합니다.")
        if self.cap: self.cap.release(); self.cap = None
        self.is_tracking = False;
        self.tracker = None

        result = self._choose_next_cctv()
        if not result:
            logging.warning("탐색 반경 내에 전환할 다른 CCTV가 없습니다.")
            return False

        dist, next_cam = result
        logging.info(f"다음 CCTV '{next_cam['id']}' (거리: {dist:.2f}km)로 전환합니다.")
        self.current_cam = next_cam
        return True

    def _detect_vehicles(self, frame):
        # ... (이전과 동일) ...
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).to(self.device).permute(2, 0, 1).float() / 255.0
        img = img.unsqueeze(0)

        t_start = time.perf_counter()
        with torch.no_grad():
            pred = self.model(img, augment=False)[0]
        t_end = time.perf_counter()
        logging.info(f"모델 추론 시간: {(t_end - t_start) * 1000:.2f}ms")

        pred = non_max_suppression(pred, self.args.conf_thres, self.args.iou_thres)[0]
        boxes = []
        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()
            for *xyxy, conf, cls in pred.cpu().numpy():
                if int(cls) in self.args.classes:
                    boxes.append(tuple(map(int, xyxy)))
        return boxes

    def _run_detection(self, frame):
        # ... (이전과 동일) ...
        if self.frame_id % self.args.interval == 0:
            self.boxes_to_draw = self._detect_vehicles(frame)
            logging.info(f"{len(self.boxes_to_draw)}개의 차량 탐지됨.")

        for (x1, y1, x2, y2) in self.boxes_to_draw: cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "Click a box to track | Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), 2)
        return frame

    def _run_tracking(self, frame):
        # ... (이전과 동일) ...
        ok, bbox = self.tracker.update(frame)
        if ok:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "TRACKING", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            logging.info("추적 실패!")
        return frame, ok

    def _log_resource_usage(self):
        """시스템 및 GPU 자원 사용량을 로깅합니다."""
        cpu_usage = self.process.cpu_percent()
        mem_usage = self.process.memory_info().rss / (1024 * 1024)  # MB
        log_msg = f"Resource Usage: CPU {cpu_usage:.2f}%, Memory {mem_usage:.2f}MB"

        if self.device.type == 'cuda':
            gpu_mem = torch.cuda.memory_allocated(self.device) / (1024 * 1024)  # MB
            log_msg += f", GPU Memory {gpu_mem:.2f}MB"

        logging.info(log_msg)

    def run(self):
        # ... (이전과 동일) ...
        if not self.model or not self.all_cams: logging.error("초기화 실패."); return 1

        for i, cam in enumerate(self.all_cams): logging.info(f"[{i}] {cam['id']}")
        try:
            idx = int(input("시작할 CCTV의 인덱스를 입력하세요: "))
            self.current_cam = self.all_cams[idx]
        except (ValueError, IndexError):
            logging.error("잘못된 인덱스입니다.");
            return 1

        cv2.namedWindow("CCTV")
        cv2.setMouseCallback("CCTV", self._on_mouse)

        try:
            while not self._shutdown_request:
                if self.cap is None:
                    self.cap = cv2.VideoCapture(self.current_cam['url'])
                    if not self.cap.isOpened():
                        logging.error(f"스트림을 열 수 없습니다: {self.current_cam['url']}")
                        if not self._switch_to_next_cam(): break
                        continue
                    logging.info(f"현재 CCTV: {self.current_cam['id']}")

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    if not self._switch_to_next_cam(): break
                    continue

                self.current_frame = frame.copy()
                now = time.time()
                self.fps = 1 / (now - self.last_fps_time) if (now - self.last_fps_time) > 0 else 0
                self.last_fps_time = now

                if self.is_tracking:
                    processed_frame, ok = self._run_tracking(self.current_frame)
                    if not ok:
                        if not self._switch_to_next_cam(): break
                        continue
                else:
                    processed_frame = self._run_detection(self.current_frame)

                cv2.putText(processed_frame, f"FPS: {self.fps:.2f}", (processed_frame.shape[1] - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("CCTV", processed_frame)

                # 주기적으로 자원 사용량 로깅
                if self.frame_id % 100 == 0:
                    self._log_resource_usage()

                if self.frame_id > 1_000_000_000: self.frame_id = 0
                self.frame_id += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("사용자 요청(q)으로 프로그램을 종료합니다.")
                    break
        finally:
            if self.cap: self.cap.release()
            cv2.destroyAllWindows()
            logging.info("리소스 정리 완료.")
        return 0


def check_float_range(value):
    # ... (이전과 동일) ...
    try:
        f = float(value);
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid float")
    if not 0.0 <= f <= 1.0: raise argparse.ArgumentTypeError(f"{value} is not between 0.0 and 1.0")
    return f


def check_positive_int(value):
    # ... (이전과 동일) ...
    try:
        i = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid integer")
    if i <= 0: raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return i


def run_tests():
    """핵심 유틸리티 함수들과 로직의 단위 테스트를 실행합니다."""
    print("--- 단위 테스트 시작 ---")

    # Haversine 테스트
    seoul = (37.5665, 126.9780);
    busan = (35.1796, 129.0756)
    dist = haversine(seoul[0], seoul[1], busan[0], busan[1])
    assert 320 < dist < 330, f"Haversine 테스트 실패: {dist}km"
    print(f"[PASS] Haversine(서울-부산) 테스트: {dist:.2f}km")

    # Argparse 유효성 검사 테스트
    assert check_float_range('0.5') == 0.5
    try:
        check_float_range('1.1'); assert False, "범위 초과값 허용"
    except argparse.ArgumentTypeError:
        print("[PASS] check_float_range(1.1) 실패 테스트")

    assert check_positive_int('10') == 10
    try:
        check_positive_int('0'); assert False, "0 허용"
    except argparse.ArgumentTypeError:
        print("[PASS] check_positive_int(0) 실패 테스트")

    # _switch_to_next_cam 로직 테스트 (Mocking)
    mock_args = MagicMock()
    mock_args.radius = 1.0

    tracker = MagicMock()
    tracker.args = mock_args
    tracker.current_cam = {'id': 'CCTV1', 'lat': 37.0, 'lon': 127.0}
    tracker.all_cams = [
        {'id': 'CCTV1', 'lat': 37.0, 'lon': 127.0},
        {'id': 'CCTV2', 'lat': 37.001, 'lon': 127.001},  # 가장 가까움
        {'id': 'CCTV3', 'lat': 37.01, 'lon': 127.01},  # 1km 내
        {'id': 'CCTV4', 'lat': 38.0, 'lon': 128.0},  # 1km 밖
    ]

    # 가장 가까운 CCTV2가 선택되어야 함
    result = VehicleTracker._choose_next_cctv(tracker)
    assert result[1]['id'] == 'CCTV2', f"_choose_next_cctv 실패: {result[1]['id']}"
    print(f"[PASS] _choose_next_cctv 테스트: {result[1]['id']} 선택됨")

    print("--- 모든 단위 테스트 통과 ---")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robust YOLOv5 CCTV Vehicle Tracker",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--test', action='store_true', help='Run unit tests and exit.')
    parser.add_argument('--api-key', type=str, help='ITS Open API Service Key (Required for normal run)')
    parser.add_argument('--weights', type=str,
                        help='Path to weights file (.pt).\nIf not specified, finds the latest in runs/train/')
    parser.add_argument('--conf-thres', type=check_float_range, default=0.4, metavar='[0.0-1.0]',
                        help='Object confidence threshold (default: 0.4)')
    parser.add_argument('--iou-thres', type=check_float_range, default=0.5, metavar='[0.0-1.0]',
                        help='IOU threshold for NMS (default: 0.5)')
    parser.add_argument('--interval', type=check_positive_int, default=5, metavar='[>0]',
                        help='Frame detection interval (default: 5)')
    parser.add_argument('--radius', type=float, default=1.0, help='Search radius for next CCTV in km (default: 1.0)')
    parser.add_argument('--classes', nargs='+', type=int, default=VehicleTracker.DEFAULT_CLASSES,
                        help=f'Class IDs to detect.\n(default: {VehicleTracker.DEFAULT_CLASSES})')
    parser.add_argument('--max-pages', type=check_positive_int, default=10, metavar='[>0]',
                        help='Maximum pages to fetch from CCTV API (default: 10)')
    args = parser.parse_args()

    exit_code = 0
    if args.test:
        run_tests()
    elif not args.api_key:
        parser.error("--api-key is required unless running with --test.")
    else:
        app = None
        try:
            app = VehicleTracker(args)
            exit_code = app.run()
        except KeyboardInterrupt:
            logging.info("프로그램 강제 종료.")
            exit_code = 130
        except Exception as e:
            logging.critical(f"처리되지 않은 최상위 예외 발생: {e}", exc_info=True)
            exit_code = 1
        finally:
            # SIGINT 핸들러를 원래대로 복구
            signal.signal(signal.SIGINT, signal.SIG_DFL)

    sys.exit(exit_code)