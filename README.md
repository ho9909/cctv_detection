# CCTV 기반 실시간 차량 추적 시스템 (Real-time Vehicle Tracking System based on Public CCTV)

# 📝 프로젝트 개요 (Overview)
> 본 프로젝트는 **대한민국 공공데이터포털(ITS 국가교통정보센터)**에서 제공하는 실시간 CCTV 영상 스트림을 활용하여, 사전 학습된 YOLOv5 모델로 차량을 탐지하고 추적하는 시스템입니다.
> 사용자는 지도상의 CCTV를 선택하여 실시간 영상을 확인할 수 있으며, 영상 속 차량을 클릭하여 추적을 시작할 수 있습니다. 만약 추적하던 차량이 현재 CCTV 화면에서 벗어나면, 시스템은 자동으로 가장 가까운 인접 CCTV(반경 1km 내)를 탐색하고 화면을 전환하여 끊김 없는 추적 경험을 제공하는 것을 목표로 합니> 다.

# ✨ 주요 기능 (Features)
> 실시간 CCTV 스트리밍: 공공데이터 API를 통해 전국 고속도로 및 국도의 CCTV 영상을 실시간으로 불러옵니다.
> AI 기반 객체 탐지: YOLOv5 모델을 사용하여 영상 내의 차량(자동차, 버스, 트럭, 오토바이 등)을 실시간으로 탐지하고 바운딩 박스로 표시합니다.
> 
> 클릭 기반 상호작용 추적: 사용자가 마우스로 특정 차량을 클릭하면, 해당 차량에 대한 추적(Tracking)을 시작합니다.
> 
> 자동 CCTV 전환: 추적 중인 차량이 화면을 벗어나면, 현재 CCTV 위치를 기준으로 가장 가까운 다음 CCTV를 자동으로 찾아 스트림을 전환합니다.
> 
> 커맨드 라인 기반 유연한 실행: API 키, 모델 가중치, 신뢰도 임계값 등 주요 설정을 터미널 인자로 유연하게 제어할 수 있습니다.
> 
> 안정적인 운영을 위한 설계:
> 
> 자원 모니터링: psutil을 통해 CPU/메모리 사용량을 주기적으로 로깅하여 장시간 운영 시 안정성을 확보합니다.
> 
> 안전한 종료: Ctrl+C 신호를 감지하여 사용 중인 모든 리소스를 안전하게 정리하고 종료합니다.
> 
> 상세 로깅: 파일 기반(tracker_session.log) 로깅 및 회전(Rotating) 기능을 통해 모든 동작 기록을 상세히 남겨 디버깅 및 성능 분석을 용이하게 합니다.
> 
> 단위 테스트: 핵심 로직(거리 계산, 인자 유효성 검사 등)에 대한 단위 테스트 코드를 포함하여 코드의 신뢰성을 보장합니다.
> 
<br />

# 🛠️ 기술 스택 (Tech Stack)
> 언어: Python 3.9+
> 
> AI/ML: PyTorch, YOLOv5
> 
> 컴퓨터 비전: OpenCV-Python
> 
> 데이터 처리: requests, psutil
> 
> 기타: argparse, unittest
<br />
# ⚙️ 설치 및 설정 (Installation & Setup)
> 본 프로젝트를 실행하기 위해 아래의 단계를 따라주세요.
>
<br />

>> ## 1. 프로젝트 클론
>>> git clone https://github.com/ho9909/cctv_detection.git
<br />
>>> cd cctv_detection
>>>
<br />

>> ## 2. YOLOv5 서브모듈 클론 및 설정
> 
>>> 프로젝트가 YOLOv5의 내부 함수를 직접 사용하므로, YOLOv5 레포지토리를 서브모듈로 클론하고 설정하는 과정이 필요합니다.
>>> // YOLOv5 서브모듈 초기화 및 클론
>>> git submodule update --init --recursive
<br />

>> ## 3. 필요 라이브러리 설치
>>> requirements.txt 파일을 통해 필요한 모든 파이썬 라이브러리를 설치합니다.
>>> pip install -r requirements.txt
>>> 참고: requirements.txt에는 psutil 등 필요한 모든 라이브러리가 포함되어 있어야 합니다.
<br />

>> ## 4. 공공데이터 API 키 발급
>>>
>>> ITS 국가교통정보센터 오픈 API 사이트에 접속하여 회원가입 및 로그인합니다.
>>>
>>>[실시간 돌발상황 CCTV] API를 찾아 활용신청을 하고, 발급받은 **서비스 키(API Key)**를 복사해둡니다.
<br />
# 5. ▶️ 실행 방법 (How to Run)
>
>터미널에서 track_vehicles.py 스크립트를 argparse 인자와 함께 실행합니다.
>
>> ## 1.가장 기본적인 실행 (자동 가중치 탐색)
>>> --api-key는 필수 인자입니다. 발급받은 본인의 API 키를 입력해주세요.
>>> python track_vehicles.py --api-key "YOUR_API_KEY_HERE"

>> ## 2.커스텀 모델 및 다른 설정으로 실행
>>> 학습시킨 커스텀 가중치 파일(best.pt)을 사용하고, 신뢰도 및 탐지 간격 등을 조절할 수 있습니다.
>>>
>>> python track_vehicles.py --api-key "YOUR_API_KEY_HERE" --weights "path/to/your/best.pt" --conf-thres 0.5 --interval 3
</br>

### 실행 인자 (Command-line Arguments)

| 인자 | 설명 | 기본값 |
| :--- | :--- | :--- |
| `--api-key` | **(필수)** ITS 오픈 API 서비스 키 | - |
| `--weights` | 사용할 `.pt` 가중치 파일 경로. 미지정 시 `runs/train/`의 최신 모델 자동 탐색 | `None` |
| `--conf-thres` | 객체 탐지 신뢰도 임계값 (0.0 ~ 1.0) | `0.4` |
| `--iou-thres` | NMS(Non-Maximum Suppression) IoU 임계값 (0.0 ~ 1.0) | `0.5` |
| `--interval` | 차량을 탐지할 프레임 간격 | `5` |
| `--radius` | 다음 CCTV를 탐색할 반경 (km) | `1.0` |
| `--classes` | 탐지할 클래스의 ID 목록 (공백으로 구분) | `2 3 5 7` |
| `--max-pages` | API로부터 가져올 최대 페이지 수 | `10` |
| `--test` | 단위 테스트를 실행하고 종료 | `False` |

> # 6. Sheets로 내보내기
>> ✅ 단위 테스트 (Unit Tests)
>>코드의 핵심 로직이 올바르게 동작하는지 확인하기 위해 단위 테스트를 실행할 수 있습니다.
>>
>> python track_vehicles.py --test
</br>
