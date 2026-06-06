# 🛠 모델 합성 및 하드웨어 변환 (Model Synthesis & Hardware Configuration)

학습된 PyTorch 모델(ResNet-lite)을 **MAX78000** 타겟 하드웨어에 배포하기 위한 변환(Synthesis) 스크립트 및 설정 파일 모음.
하드웨어 가속 여부에 따라 **MCU (CMSIS-NN)** 와 **NPU (Hardware Accelerator)** 두 경로로 구성됨.

---

## 📂 디렉토리 구조

- `mcu/` : QAT 가중치 → ARM CMSIS-NN 포맷 변환 커스텀 스크립트
- `npu/` : MAX78000 CNN 가속기 매핑용 하드웨어 인식 설정 파일 (YAML)

---

## 1. MCU Synthesis (CMSIS-NN 가중치 추출)

ARM Cortex-M4 코어에서 순수 소프트웨어 추론 수행을 위해 양자화 가중치를 CMSIS-NN 규격으로 변환함.

- 1D Conv 가중치 → 2D(H=1) 포맷으로 Transpose
- 하드웨어 정렬을 위한 Dummy Bias 배열 강제 주입 로직 포함

### 🚀 사용 방법

```bash
cd mcu

# 1. 체크포인트 내부 PyTorch 레이어 이름(Key) 확인
python check_keys.py --checkpoint path/to/qat_best-q.pth.tar

# 2. 가중치 추출 및 C-header (mcu_weights.h) 생성
python export_all_weights.py --checkpoint path/to/qat_best-q.pth.tar --output mcu_weights.h
```

---

## 2. NPU Synthesis (하드웨어 매핑 및 최적화)

Maxim 툴체인으로 신경망을 MAX78000 NPU 가속기에 합성함.
`npu/` 내부 YAML 파일들은 프로세서 할당 및 SRAM 메모리 로드 밸런싱 실험(Model Sweep) 결과물.

### 🧪 실험 테마 및 주요 설정 파일

**1. 경량화 베이스라인 (`kws20_hwc_c32.yaml`)**

- 채널 수에 맞춰 32-bit 프로세서 마스크를 엄격히 적용
- 나머지 절반 프로세서를 유휴 상태로 두어 전력 최적화

**2. 네트워크 깊이(Depth) 확장 테스트 (`kws20_hwc_deep_d4_c64.yaml` 등)**

- 레이어가 깊어질수록 발생하는 메모리 충돌 방지 목적
- 4개의 SRAM 구역(`0x0000` → `0x2000` → `0x4000` → `0x6000`)을 순회하는 Ping-Pong 메모리 라우팅 기법 적용

**3. 물리적 하드웨어 한계 테스트 (`kws20_hwc_c96.yaml` 등)**

- NPU 합성 엔진의 절대 상한선 검증용 의도적 실패 케이스
- 채널 확장 시 프로세서당 432KB SRAM 한계 초과 과정을 에러 로그로 증명

### 🚀 NPU 합성 실행 방법

> ⚠️ 사전에 로컬 PC에 Maxim SDK 툴체인 설치 필요

```bash
python ai8xize.py \
  --network resnet_lite \
  --config networks/kws20_hwc_baseline_c64.yaml \
  --checkpoint path/to/qat_best-q.pth.tar \
  --device MAX78000 \
  --compact-data
```