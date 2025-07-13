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
import psutil
from logging.handlers import RotatingFileHandler
from unittest.mock import MagicMock

# YOLOv5 레포지토리 함수 import (pip install -e . 실행 후 가능)
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords

def setup_logging():
    """파일 회전 및 중복 방지 기능이 포함된 로거를 설정합니다."""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
    root_logger = logging.getLogger()
    # 내가 등록하려는 핸들러 종류가 없을 때만 설정
    if not any(isinstance(h, (logging.handlers.RotatingFileHandler, logging.StreamHandler)) for h in root_logger.handlers):
        root_logger.setLevel(logging.INFO)
        # 파일 회전 핸들러 (10MB, 5개 백업)
        file_handler = RotatingFileHandler('tracker_session.log', maxBytes=10**7, backupCount=5, encoding='utf-8')
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        root_logger.addHandler(console_handler)

def haversine(lat1, lon1, lat2, lon2):
    """두 위경도 지점 간의 거리를 km 단위로 계산합니다."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class VehicleTracker:
    """
    CCTV 차량 추적 애플리케이션의 모든 로직과 상태를 관리하는 클래스.
    """
    DEFAULT_CLASSES = [2, 3, 5, 7] # car, motorbike, bus, truck

    def __init__(self, args):
        setup_logging()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 종료 신호 처리를 위한 플래그 설정
        self._shutdown_request = False
        signal.signal(signal.SIGINT, self._graceful_shutdown)
        
        # 초기화 및 검증
        self.model = self._load_model(self.args.weights)
        self._validate_class_ids()
        self.all_cams = self._fetch_cctv_list()
        
        # 상태 변수 초기화
        self.current_cam = None
        self.cap = None
        self.tracker = None
        self.is_tracking = False
        self.boxes_to_draw = []
        self.frame_id = 0
        self.current_frame = None
        self.last_fps_time = time.time()
        self.fps = 0
        self.process = psutil.Process(os.getpid())

    def _graceful_shutdown(self, signum, frame):
        """SIGINT(Ctrl+C) 신호를 처리하여 안전한 종료를 요청합니다."""
        if not self._shutdown_request:
            logging.warning("종료 신호(Ctrl+C) 수신. 안전하게 종료를 시도합니다...")
            self._shutdown_request = True

    def _load_model(self, weights_path):
        """지정된 경로의 YOLOv5 모델을 로드하고 검증합니다."""
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
            logging.critical(f"모델 로드 중 심각한 오류 발생: {e}", exc_info=True)
            return None
            
    def _validate_class_ids(self):
        """사용자가 입력한 클래스 ID의 유효성을 검증하고, 유효하지 않으면 기본값으로 복원합니다."""
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
        """ITS API를 통해 CCTV 목록을 가져옵니다."""
        cams = []
        page = 1
        logging.info("ITS API에서 CCTV 목록을 가져오는 중...")
        while page <= self.args.max_pages:
            url = f"http://openapi.its.go.kr:8081/api/NCCTVInfo?key={self.args.api_key}&ReqType=2&type=ex&pageNo={page}"
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                root = ET.fromstring(r.content)
                items = root.findall('data')
                if not items:
                    logging.info(f"총 {page-1} 페이지에서 CCTV 정보 수집 완료.")
                    break
                for it in items:
                    u_node = it.find('cctvurl')
                    if u_node is not None and u_node.text:
                        cams.append({
                            'id': it.find('cctvname').text,
                            'lat': float(it.find('coordy').text), 
                            'lon': float(it.find('coordx').text),
                            'url': u_node.text
                        })
                page += 1
            except requests.exceptions.RequestException as e:
                logging.error(f"CCTV API 요청 오류 (Page {page}): {e}")
                break
            except ET.ParseError as e:
                logging.error(f"CCTV API 응답 파싱 오류 (Page {page}): {e}")
                break
        if page > self.args.max_pages:
            logging.warning(f"최대 페이지({self.args.max_pages})에 도달하여 수집을 중단합니다.")
        return cams

    def _on_mouse(self, event, x, y, flags, param):
        """마우스 클릭 이벤트를 처리하여 추적을 시작합니다."""
        if event == cv2.EVENT_LBUTTONDOWN and not self.is_tracking:
            for (x1, y1, x2, y2) in self.boxes_to_draw:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.tracker = cv2.TrackerCSRT_create()
                    self.tracker.init(self.current_frame, (x1, y1, x2 - x1, y2 - y1))
                    self.is_tracking = True
                    logging.info(f"추적 시작: Box({x1},{y1},{x2},{y2})")
                    break

    def _choose_next_cctv(self):
        """현재 CCTV 반경 내에서 가장 가까운 다음 CCTV를 선택합니다."""
        candidates = []
        if not self.current_cam: return None
        for c in self.all_cams:
            if c['id'] == self.current_cam['id']: continue
            d = haversine(self.current_cam['lat'], self.current_cam['lon'], c['lat'], c['lon'])
            if d <= self.args.radius:
                candidates.append((d, c))
        if not candidates:
            return None
        return sorted(candidates, key=lambda x: x[0])[0]

    def _switch_to_next_cam(self):
        """CCTV 전환 로직을 통합하여 처리합니다."""
        logging.info("다음 CCTV로 전환을 시도합니다.")
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_tracking = False
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
        """주어진 프레임에서 차량을 탐지하고, 추론 시간을 로깅합니다."""
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
        """탐지 모드 로직을 실행합니다."""
        if self.frame_id % self.args.interval == 0:
            self.boxes_to_draw = self._detect_vehicles(frame)
            logging.info(f"{len(self.boxes_to_draw)}개의 차량 탐지됨.")
        
        for (x1, y1, x2, y2) in self.boxes_to_draw:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "Click a box to track | Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return frame

    def _run_tracking(self, frame):
        """추적 모드 로직을 실행합니다."""
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
        cpu_usage = self.process.cpu_percent(interval=None)
        mem_usage = self.process.memory_info().rss / (1024 * 1024) # MB
        log_msg = f"Resource Usage: CPU {cpu_usage:.2f}%, Memory {mem_usage:.2f}MB"
        
        if self.device.type == 'cuda':
            gpu_mem = torch.cuda.memory_allocated(self.device) / (1024 * 1024) # MB
            log_msg += f", GPU Memory {gpu_mem:.2f}MB"
        
        logging.info(log_msg)

    def run(self):
        """애플리케이션의 메인 루프를 실행하고 종료 코드를 반환합니다."""
        if not self.model:
            logging.critical("모델이 로드되지 않았습니다. 프로그램을 시작할 수 없습니다.")
            return 1
        if not self.all_cams:
            logging.critical("CCTV 목록을 가져오지 못했습니다. API 키나 네트워크를 확인해주세요.")
            return 1

        for i, cam in enumerate(self.all_cams):
            logging.info(f"[{i}] {cam['id']}")
        try:
            idx = int(input("시작할 CCTV의 인덱스를 입력하세요: "))
            self.current_cam = self.all_cams[idx]
        except (ValueError, IndexError):
            logging.error("잘못된 인덱스입니다.")
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
                time_diff = now - self.last_fps_time
                self.fps = 1 / time_diff if time_diff > 0 else 0
                self.last_fps_time = now

                if self.is_tracking:
                    processed_frame, ok = self._run_tracking(self.current_frame)
                    if not ok:
                        if not self._switch_to_next_cam(): break
                        continue
                else:
                    processed_frame = self._run_detection(self.current_frame)
                
                cv2.putText(processed_frame, f"FPS: {self.fps:.2f}", (processed_frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("CCTV", processed_frame)
                
                if self.frame_id % 100 == 0:
                    self._log_resource_usage()

                if self.frame_id > 1_000_000_000: self.frame_id = 0
                self.frame_id += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("사용자 요청(q)으로 프로그램을 종료합니다.")
                    break
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            logging.info("리소스 정리 완료.")
        return 0

def check_float_range(value):
    """argparse를 위한 float 범위 검사 함수."""
    try:
        f = float(value)
        if 0.0 <= f <= 1.0: return f
    except ValueError: pass
    raise argparse.ArgumentTypeError(f"입력값 '{value}'는 0.0과 1.0 사이의 실수가 아닙니다.")

def check_positive_int(value):
    """argparse를 위한 양의 정수 검사 함수."""
    try:
        i = int(value)
        if i > 0: return i
    except ValueError: pass
    raise argparse.ArgumentTypeError(f"입력값 '{value}'는 양의 정수가 아닙니다.")

def run_tests():
    """핵심 유틸리티 함수들과 로직의 단위 테스트를 실행합니다."""
    print("--- 단위 테스트 시작 ---")
    # Haversine 테스트 (서울-부산)
    seoul = (37.5665, 126.9780)
    busan = (35.1796, 129.0756)
    dist = haversine(seoul[0], seoul[1], busan[0], busan[1])
    assert 320 < dist < 330, f"Haversine 테스트 실패: {dist}km"
    print(f"[PASS] Haversine(서울-부산) 테스트: {dist:.2f}km")

    # Argparse 유효성 검사 테스트
    assert check_float_range('0.5') == 0.5
    try: check_float_range('1.1'); assert False, "범위 초과값 허용"
    except argparse.ArgumentTypeError: print("[PASS] check_float_range(1.1) 실패 테스트")
    
    assert check_positive_int('10') == 10
    try: check_positive_int('0'); assert False, "0 허용"
    except argparse.ArgumentTypeError: print("[PASS] check_positive_int(0) 실패 테스트")

    # _switch_to_next_cam 로직 테스트 (Mocking)
    mock_args = MagicMock()
    mock_args.radius = 1.0
    
    tracker = MagicMock()
    tracker.args = mock_args
    tracker.current_cam = {'id': 'CCTV1', 'lat': 37.0, 'lon': 127.0}
    tracker.all_cams = [
        {'id': 'CCTV1', 'lat': 37.0, 'lon': 127.0},
        {'id': 'CCTV2', 'lat': 37.001, 'lon': 127.001}, # 가장 가까움
        {'id': 'CCTV3', 'lat': 37.01, 'lon': 127.01},   # 1km 내
        {'id': 'CCTV4', 'lat': 38.0, 'lon': 128.0},     # 1km 밖
    ]
    
    result = VehicleTracker._choose_next_cctv(tracker)
    assert result[1]['id'] == 'CCTV2', f"_choose_next_cctv 실패: {result[1]['id']}"
    print(f"[PASS] _choose_next_cctv 테스트: {result[1]['id']} 선택됨")

    print("--- 모든 단위 테스트 통과 ---")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Robust YOLOv5 CCTV Vehicle Tracker",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--test', action='store_true', 
        help='단위 테스트를 실행하고 종료합니다.'
    )
    parser.add_argument(
        '--api-key', type=str, 
        help='ITS Open API 서비스 키 (일반 실행 시 필수)'
    )
    parser.add_argument(
        '--weights', type=str, 
        help='사용할 .pt 가중치 파일 경로.\n미지정 시 runs/train/ 폴더의 최신 모델을 자동 탐색합니다.'
    )
    parser.add_argument(
        '--conf-thres', type=check_float_range, default=0.4, metavar='[0.0-1.0]', 
        help='객체 탐지 신뢰도 임계값 (기본값: 0.4)'
    )
    parser.add_argument(
        '--iou-thres', type=check_float_range, default=0.5, metavar='[0.0-1.0]', 
        help='NMS(Non-Maximum Suppression) IoU 임계값 (기본값: 0.5)'
    )
    parser.add_argument(
        '--interval', type=check_positive_int, default=5, metavar='[>0]', 
        help='차량을 탐지할 프레임 간격 (기본값: 5)'
    )
    parser.add_argument(
        '--radius', type=float, default=1.0, 
        help='다음 CCTV를 탐색할 반경(km) (기본값: 1.0)'
    )
    parser.add_argument(
        '--classes', nargs='+', type=int, default=VehicleTracker.DEFAULT_CLASSES, 
        help=f'탐지할 클래스의 ID 목록 (공백으로 구분).\n(기본값: {VehicleTracker.DEFAULT_CLASSES})'
    )
    parser.add_argument(
        '--max-pages', type=check_positive_int, default=10, metavar='[>0]', 
        help='ITS API로부터 가져올 최대 페이지 수 (기본값: 10)'
    )
    
    args = parser.parse_args()
    
    exit_code = 0
    if args.test:
        run_tests()
    elif not args.api_key:
        parser.error("--api-key 인자는 --test 옵션 없이 실행할 때 필수입니다.")
    else:
        app = None
        try:
            app = VehicleTracker(args)
            exit_code = app.run()
        except KeyboardInterrupt:
            logging.info("프로그램 강제 종료됨.")
            exit_code = 130 # SIGINT에 대한 표준 종료 코드
        except Exception as e:
            logging.critical(f"처리되지 않은 최상위 예외 발생: {e}", exc_info=True)
            exit_code = 1
        finally:
            # 기본 SIGINT 핸들러 복원
            signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    logging.info(f"프로그램 종료. 종료 코드: {exit_code}")
    sys.exit(exit_code)
