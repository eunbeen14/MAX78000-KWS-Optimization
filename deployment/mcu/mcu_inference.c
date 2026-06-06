"""
=============================================================================
[MCU Deployment: CMSIS-NN Inference Engine (Dynamic Depth)]
Description: 
A pure software implementation of the Custom ResNet-lite architecture using 
the ARM CMSIS-NN library. Designed to benchmark baseline latency on the 
Cortex-M4 core.

Features:
- Dynamic Depth Compilation: Uses the RESNET_DEPTH macro (injected via Makefile)
  to conditionally compile extra convolution layers (Depth 0, 2, or 4).
- Custom 1D Pooling configuration for Time-series audio data.
- Strict SRAM memory management (Ping-pong buffers) to prevent overflow.
=============================================================================
"""
#include "mcu_inference.h"
#include "mcu_weights.h"
#include <string.h> // For memcpy

static int8_t buffer_A[MAX_FEATURE_MAP_SIZE];
static int8_t buffer_B[MAX_FEATURE_MAP_SIZE];
static int8_t buffer_shortcut[MAX_FEATURE_MAP_SIZE];
static int8_t ctx_buffer[MAX_FEATURE_MAP_SIZE];

// [Macro Check] Ensure RESNET_DEPTH is defined (default to 0 if not passed via Makefile)
#ifndef RESNET_DEPTH
#define RESNET_DEPTH 0
#endif

void run_mcu_resnet_lite(const int8_t* input_data, int8_t* final_output)
{
    cmsis_nn_context ctx;
    ctx.buf = ctx_buffer;
    ctx.size = sizeof(ctx_buffer);

    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;
    cmsis_nn_pool_params pool_params;
    
    // Custom 1D Pooling configuration (Height=1, Width=2)
    cmsis_nn_dims pool_filter_dims;
    pool_filter_dims.n = 1; pool_filter_dims.h = 1; pool_filter_dims.w = 2; pool_filter_dims.c = 1;

    // Dummy quantization parameters for pure latency benchmarking
    int32_t mult_array[128];
    int32_t shift_array[128];
    for(int i = 0; i < 128; i++) {
        mult_array[i] = 1;
        shift_array[i] = 0;
    }
    quant_params.multiplier = mult_array;
    quant_params.shift = shift_array;
    bias_dims.n = 1; bias_dims.h = 1; bias_dims.w = 1;

    // ==========================================
    // LAYER 0: voice_conv1
    // ==========================================
    conv_params.padding.w = 0; conv_params.padding.h = 0;
    conv_params.stride.w = 1; conv_params.stride.h = 1;
    conv_params.dilation.w = 1; conv_params.dilation.h = 1;
    conv_params.activation.min = 0; // ReLU
    conv_params.activation.max = 127;
    
    input_dims.n = 1; input_dims.h = 1; input_dims.w = 128; input_dims.c = 128;
    filter_dims.n = 96; filter_dims.h = 1; filter_dims.w = 1; filter_dims.c = 128;
    bias_dims.c = 96;
    output_dims.n = 1; output_dims.h = 1; output_dims.w = 128; output_dims.c = 96;

    arm_convolve_s8(&ctx, &conv_params, &quant_params, &input_dims, input_data, 
                    &filter_dims, voice_conv1_wt, &bias_dims, voice_conv1_bias, 
                    &output_dims, buffer_A);

    // ==========================================
    // LAYER 1: voice_conv2
    // ==========================================
    input_dims = output_dims; // 128
    filter_dims.n = 96; filter_dims.w = 3; filter_dims.c = 96;
    bias_dims.c = 96;
    output_dims.w = input_dims.w - 3 + 1; output_dims.c = 96; // 126

    arm_convolve_s8(&ctx, &conv_params, &quant_params, &input_dims, buffer_A, 
                    &filter_dims, voice_conv2_wt, &bias_dims, voice_conv2_bias, 
                    &output_dims, buffer_B);

    // =======================================================
    // 🔥 [DYNAMIC COMPILATION] EXTRA DEPTH BLOCKS
    // Compiled conditionally based on RESNET_DEPTH macro.
    // =======================================================
#if (RESNET_DEPTH > 0)
    conv_params.padding.w = 1; // Maintain width (126) for deep sequential layers
    
    // Array of weight pointers to optimize the unrolling loop
#if (RESNET_DEPTH == 2)
    const int8_t* extra_wt[] = {extra_convs_0_wt, extra_convs_1_wt};
    const int32_t* extra_bias[] = {extra_convs_0_bias, extra_convs_1_bias};
#elif (RESNET_DEPTH == 4)
    const int8_t* extra_wt[] = {extra_convs_0_wt, extra_convs_1_wt, extra_convs_2_wt, extra_convs_3_wt};
    const int32_t* extra_bias[] = {extra_convs_0_bias, extra_convs_1_bias, extra_convs_2_bias, extra_convs_3_bias};
#endif

    for(int i = 0; i < RESNET_DEPTH; i++) {
        // 1. Preserve Shortcut
        memcpy(buffer_shortcut, buffer_B, 126 * 96);
        
        // 2. Conv without ReLU
        conv_params.activation.min = -128; 
        input_dims.w = 126; input_dims.c = 96;
        filter_dims.n = 96; filter_dims.w = 3; filter_dims.c = 96;
        output_dims.w = 126; output_dims.c = 96;
        
        arm_convolve_s8(&ctx, &conv_params, &quant_params, &input_dims, buffer_B, 
                        &filter_dims, extra_wt[i], &bias_dims, extra_bias[i], 
                        &output_dims, buffer_A);
                        
        // 3. Add with ReLU Activation
        arm_elementwise_add_s8(
            buffer_A, buffer_shortcut, 0, 1, 0, 0, 1, 0, 0, 
            buffer_B, 0, 1, 0, 0, 127, (126 * 96) // ReLU bounds
        );
    }
#endif // (RESNET_DEPTH > 0)

    // ==========================================
    // LAYER 2: voice_conv3 (MaxPool -> Conv)
    // ==========================================
    conv_params.padding.w = 0; // Restore padding
    pool_params.stride.w = 2; pool_params.stride.h = 1;
    pool_params.padding.w = 0; pool_params.padding.h = 0;
    pool_params.activation.min = -128; pool_params.activation.max = 127;
    
    input_dims.w = 126; input_dims.c = 96;
    output_dims.w = input_dims.w / 2; // 63
    
    arm_max_pool_s8(&ctx, &pool_params, &input_dims, buffer_B, &pool_filter_dims, &output_dims, buffer_A);
    memcpy(buffer_shortcut, buffer_A, 63 * 96); // Save shortcut
    
    conv_params.activation.min = -128; // None
    input_dims = output_dims; // 63
    filter_dims.n = 64; filter_dims.w = 3; filter_dims.c = 96;
    bias_dims.c = 64;
    output_dims.w = input_dims.w - 3 + 1; // 61
    output_dims.c = 64;
    
    arm_convolve_s8(&ctx, &conv_params, &quant_params, &input_dims, buffer_A, 
                    &filter_dims, voice_conv3_wt, &bias_dims, voice_conv3_bias, 
                    &output_dims, buffer_B);

    // ==========================================
    // LAYER 3: shortcut
    // ==========================================
    input_dims.w = 63; input_dims.c = 96;
    arm_convolve_s8(&ctx, &conv_params, &quant_params, &input_dims, buffer_shortcut, 
                    &filter_dims, shortcut_wt, &bias_dims, shortcut_bias, 
                    &output_dims, buffer_A);

    // ==========================================
    // LAYER 4: residual_add
    // ==========================================
    arm_elementwise_add_s8(
        buffer_B, buffer_A, 0, 1, 0, 0, 1, 0, 0, 
        buffer_B, 0, 1, 0, 0, 127, (61 * 64) // Output buffer is B, with ReLU
    );
    
    // ==========================================
    // LAYER 5: voice_conv4
    // ==========================================
    conv_params.activation.min = 0; // ReLU
    input_dims = output_dims; // 61
    filter_dims.n = 48; filter_dims.w = 3; filter_dims.c = 64;
    bias_dims.c = 48;
    output_dims.w = input_dims.w - 3 + 1; // 59
    output_dims.c = 48;
    
    arm_convolve_s8(&ctx, &conv_params, &quant_params, &input_dims, buffer_B, 
                    &filter_dims, voice_conv4_wt, &bias_dims, voice_conv4_bias, 
                    &output_dims, buffer_A);

    // ==========================================
    // LAYER 6: kws_conv1 (MaxPool -> Conv)
    // ==========================================
    input_dims = output_dims; // 59
    output_dims.w = input_dims.w / 2; // 29
    
    arm_max_pool_s8(&ctx, &pool_params, &input_dims, buffer_A, &pool_filter_dims, &output_dims, buffer_B);
    
    input_dims = output_dims; // 29
    filter_dims.n = 64; filter_dims.w = 3; filter_dims.c = 48;
    bias_dims.c = 64;
    output_dims.w = input_dims.w - 3 + 1; // 27
    output_dims.c = 64;
    
    arm_convolve_s8(&ctx, &conv_params, &quant_params, &input_dims, buffer_B, 
                    &filter_dims, kws_conv1_wt, &bias_dims, kws_conv1_bias, 
                    &output_dims, buffer_A);

    // ==========================================
    // LAYER 7: kws_conv2 
    // ==========================================
    input_dims = output_dims; // 27
    filter_dims.n = 96; filter_dims.w = 3; filter_dims.c = 64;
    bias_dims.c = 96;
    output_dims.w = input_dims.w - 3 + 1; // 25
    output_dims.c = 96;
    
    arm_convolve_s8(&ctx, &conv_params, &quant_params, &input_dims, buffer_A, 
                    &filter_dims, kws_conv2_wt, &bias_dims, kws_conv2_bias, 
                    &output_dims, buffer_B);

    // ==========================================
    // LAYER 8: kws_conv3 (AvgPool -> Conv)
    // ==========================================
    input_dims = output_dims; // 25
    output_dims.w = input_dims.w / 2; // 12
    
    arm_avgpool_s8(&ctx, &pool_params, &input_dims, buffer_B, &pool_filter_dims, &output_dims, buffer_A);
    
    input_dims = output_dims; // 12
    filter_dims.n = 96; filter_dims.w = 3; filter_dims.c = 96;
    output_dims.w = input_dims.w - 3 + 1; // 10
    
    arm_convolve_s8(&ctx, &conv_params, &quant_params, &input_dims, buffer_A, 
                    &filter_dims, kws_conv3_wt, &bias_dims, kws_conv3_bias, 
                    &output_dims, buffer_B);

    // ==========================================
    // LAYER 9: kws_conv4 (MaxPool -> Conv)
    // ==========================================
    input_dims = output_dims; // 10
    output_dims.w = input_dims.w / 2; // 5
    
    arm_max_pool_s8(&ctx, &pool_params, &input_dims, buffer_B, &pool_filter_dims, &output_dims, buffer_A);
    
    input_dims = output_dims; // 5
    filter_dims.n = 64; filter_dims.w = 3; filter_dims.c = 96;
    bias_dims.c = 64;
    output_dims.w = input_dims.w - 3 + 1; // 3
    output_dims.c = 64;
    
    arm_convolve_s8(&ctx, &conv_params, &quant_params, &input_dims, buffer_A, 
                    &filter_dims, kws_conv4_wt, &bias_dims, kws_conv4_bias, 
                    &output_dims, buffer_B);

    // ==========================================
    // LAYER 10: Fully Connected (FC)
    // ==========================================
    cmsis_nn_fc_params fc_params;
    fc_params.input_offset = 0; fc_params.filter_offset = 0; fc_params.output_offset = 0;
    fc_params.activation.min = -128; fc_params.activation.max = 127;

    cmsis_nn_dims fc_input_dims;
    fc_input_dims.n = 1; fc_input_dims.h = 1; fc_input_dims.w = 1; 
    fc_input_dims.c = output_dims.w * output_dims.c; // 3 * 64 = 192

    cmsis_nn_dims fc_filter_dims;
    fc_filter_dims.n = NUM_CLASSES; fc_filter_dims.h = 1; fc_filter_dims.w = 1; 
    fc_filter_dims.c = fc_input_dims.c;

    cmsis_nn_dims fc_output_dims;
    fc_output_dims.n = 1; fc_output_dims.h = 1; fc_output_dims.w = 1; 
    fc_output_dims.c = NUM_CLASSES;

    cmsis_nn_per_tensor_quant_params fc_quant_params;
    fc_quant_params.multiplier = 1;
    fc_quant_params.shift = 0;

    arm_fully_connected_s8(&ctx, &fc_params, &fc_quant_params, &fc_input_dims, buffer_B, 
                           &fc_filter_dims, fc_wt, &bias_dims, fc_bias, 
                           &fc_output_dims, final_output);
}