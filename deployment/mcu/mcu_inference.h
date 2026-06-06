#ifndef MCU_INFERENCE_H
#define MCU_INFERENCE_H

#include <stdint.h>
#include "arm_nnfunctions.h"

#define NUM_CLASSES 21
#define MAX_FEATURE_MAP_SIZE (16 * 1024) 

/**
 * @brief MCU(Cortex-M4F) 전용 ResNet-Lite 전체 모델 추론 함수
 * @param input_data     정규화 및 INT8 양자화가 완료된 입력 데이터 배열 (크기: 입력 채널 * 길이)
 * @param final_output   최종 클래스별 예측 결과가 저장될 배열 (크기: NUM_CLASSES)
 */
void run_mcu_resnet_lite(const int8_t* input_data, int8_t* final_output);

#endif // MCU_INFERENCE_H