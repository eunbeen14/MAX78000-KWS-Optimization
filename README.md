# 🚀 MAX78000 Hardware-Aware Optimization for Time-Series AI

> 경희대학교 컴퓨터공학부 26-1 졸업프로젝트 

극저전력 MCU인 **MAX78000** 환경에서 시계열 음성 데이터(KWS: Keyword Spotting) 추론을 위한   알고리즘 설계부터 임베디드 NPU 배포까지의 **End-to-End 최적화 파이프라인** 구현물이다.

단순 데모 실행을 넘어, 제한된 하드웨어 자원(SRAM, Processor)의 한계를 직접 분석하고  **메모리 로드 밸런싱** 및 **Cortex-M4 메모리 액세스 최적화** 를 적용하여 추론 지연 시간(Latency) 병목을 개선한다.

---

## 🎯 Project Highlights

### Hardware-Algorithm Co-Design
- DS-CNN, TCN-lite 등 다양한 경량 모델을 설계했으나, NPU의 하드와이어링(Hardwiring) 및 패딩 제한(Padding Limit)으로 인한 Mismatch를 실증 분석한다.
- 최종적으로 **ResNet-lite (Shortcut 적용)** 모델을 채택 및 최적화한다.

### Extreme Memory Management (Model Sweep)
- SRAM 한계를 돌파하기 위해, 채널/깊이 확장에 따른 **Ping-Pong 메모리 라우팅** (0x0000 ↔ 0x4000)을 적용한다.
- **프로세서 마스킹(Processor Masking)** 기법을 활용한 YAML 하드웨어 매핑을 구현한다.

### Firmware-Level Bottleneck Optimization
- NPU 추론 시 Cortex-M4 코어에서 발생하는 데이터 포맷팅 병목(Data Formatting Overhead)을 해소한다.
- 기존 `AddTranspose` 함수의 `%` 연산을 제거하고 **32-bit 포인터 캐스팅 + 비트 시프트(Bitwise Shift) 기반 Unrolling** 최적화를 구현한다.

### Software (MCU) vs Hardware (NPU) Profiling
- ARM CMSIS-NN 라이브러리를 활용해 순수 MCU 기반 추론 펌웨어를 직접 구현한다.
- NPU 가속기 도입에 따른 성능 향상폭(Speed-up)을 **마이크로초(μs) 단위로 프로파일링**한다.

---

## 📂 Repository Structure

파이프라인 흐름은 다음과 같다.

```
MAX78000-KWS-Optimization/
│
├── training/                # PyTorch 모델 설계 및 양자화 학습 (QAT)
│   ├── models/                # 스크래치 설계 모델 구조
│   └── train.py               # Maxim SDK 제공 학습 스크립트 활용
│
├── synthesis/               # 모델 합성 및 타겟 하드웨어 매핑
│   ├── mcu/                   # C-Array 추출 스크립트
│   └── npu/                   # MAX78000 NPU용 커스텀 YAML 매핑 파일
│
└── deployment/              # 실제 타겟 보드 배포 및 추론
    ├── mcu/                   # ARM Cortex-M4 순수 소프트웨어 추론 코드 
    └── npu_accelerated/       # NPU 가속기 추론 및 32-bit Transpose     
                                 Unrolling 최적화 메인 코드
```

---

## 🛠 How to Build & Reproduce

> 무거운 종속성 라이브러리(TFT, BSP 등)를 제거한 **클린(Clean) 버전**을 제공한다.  
> 빌드 및 보드 플래싱을 위해서는 **Maxim SDK (MSDK)** 환경이 필요하다.

### Step 1. Prepare Dependencies (SDK)

| 항목 | 설명 |
|------|------|
| NPU 툴체인 | 학습 모델 합성(`synthesis`)을 위한 **Maximizer 툴체인**이 필요하다. |
| 펌웨어 빌드 | SDK 내 기본 유틸리티(Utility), 디스플레이(TFT), CMSIS 라이브러리를 로컬 경로에 포함한다. (`project.mk`를 통해 링킹) |

### Step 2. Synthesis

새로운 모델 가중치를 하드웨어 언어로 합성할 경우 `synthesis` 폴더의 스크립트를 사용한다.

**MCU용**
```bash
python export_all_weights.py
# → 펌웨어용 mcu_weights.h 획득
```

**NPU용**
```
izer 환경에서 커스텀 .yaml 파일 지정 → 변환 수행
→ cnn.c, cnn.h, weights.h 획득
```

> 기본 베이스라인 파일은 `deployment/npu_accelerated/` 에 포함되어 있다.

### Step 3. Build & Flash

`deployment` 내의 타겟 폴더로 이동 후 Make로 빌드한다.  
*(Step 2에서 생성된 파일들을 해당 폴더에 덮어씌워야 함.)*

**MCU**
```bash
cd deployment/mcu
make RESNET_DEPTH=0  # Depth를 변경하여 동적 매크로 빌드 가능
make flash
```

**NPU Accelerated**
```bash
cd deployment/npu_accelerated
make
make flash
```


---
 
## 📜 Acknowledgments & References
 
본 프로젝트는 아래 오픈소스 프레임워크, SDK, 라이브러리를 기반으로 구현되었다.  
각 SDK의 원본 저작권 및 라이선스는 해당 소유자에게 있다.
 
- **[Maxim Integrated AI8X Toolchain](https://github.com/analogdevicesinc/ai8x-synthesis):** 신경망 합성 컴파일러(`izer`) 및 PyTorch 양자화 인식 학습(QAT) 프레임워크([`ai8x-training`](https://github.com/analogdevicesinc/ai8x-training))를 모델 양자화 및 NPU 매핑에 활용한다. (Apache 2.0 License)
- **[Analog Devices MSDK](https://github.com/analogdevicesinc/msdk):** MAX78000 마이크로컨트롤러의 핵심 하드웨어 드라이버 및 BSP를 제공한다.
- **[ARM CMSIS-NN](https://github.com/ARM-software/CMSIS-NN):** Cortex-M4 코어에서의 소프트웨어 기반 베이스라인 추론에 ARM이 제공하는 최적화된 CMSIS-NN 라이브러리를 활용한다.
- **Custom Contributions:** 본 저장소 내의 독자적 최적화 기법—*32-bit Memory Unrolling (`AddTranspose` bypass)*, *Dynamic Depth C-array generation*, *Hardware-Aware Memory Load Balancing (YAML)*—은 본 연구/포트폴리오 프로젝트를 위해 독립적으로 개발되었다.