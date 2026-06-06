/**************************************************************************************************
* Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
*
* Maxim Integrated Products, Inc. Default Copyright Notice:
* https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
**************************************************************************************************/

/*
 * This header file was automatically @generated for the kws20 network from a template.
 * Please do not edit; instead, edit the template and regenerate.
 */

#ifndef __CNN_H__
#define __CNN_H__

#include <stdint.h>
typedef int32_t q31_t;
typedef int16_t q15_t;

/* Return codes */
#define CNN_FAIL 0
#define CNN_OK 1

/*
  SUMMARY OF OPS
  Hardware: 7,127,328 ops (7,078,848 macc; 42,336 comp; 6,144 add; 0 mul; 0 bitwise)
    Layer 0 (voice_conv1_Conv_8): 1,585,152 ops (1,572,864 macc; 12,288 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1 (voice_conv2_Conv_6): 1,732,416 ops (1,714,176 macc; 18,240 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2 (voice_conv3_Conv_6): 1,105,920 ops (1,105,920 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3 (shortcut_Conv_6): 1,105,920 ops (1,105,920 macc; 0 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4 (residual_add): 3,840 ops (0 macc; 0 comp; 3,840 add; 0 mul; 0 bitwise)
    Layer 5 (voice_conv4_Conv_6): 537,312 ops (534,528 macc; 2,784 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6 (kws_conv1_Conv_6): 253,344 ops (248,832 macc; 4,512 comp; 0 add; 0 mul; 0 bitwise)
    Layer 7 (kws_conv2_Conv_6): 463,200 ops (460,800 macc; 2,400 comp; 0 add; 0 mul; 0 bitwise)
    Layer 8 (kws_conv3_Conv_6): 279,744 ops (276,480 macc; 960 comp; 2,304 add; 0 mul; 0 bitwise)
    Layer 9 (kws_conv4_Conv_6): 56,448 ops (55,296 macc; 1,152 comp; 0 add; 0 mul; 0 bitwise)
    Layer 10 (fc_MatMul_3): 4,032 ops (4,032 macc; 0 comp; 0 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 163,776 bytes out of 442,368 bytes total (37.0%)
  Bias memory:   0 bytes out of 2,048 bytes total (0.0%)
*/

/* Number of outputs for this network */
#define CNN_NUM_OUTPUTS 11

/* Port pin actions used to signal that processing is active */

#define CNN_START LED_On(1)
#define CNN_COMPLETE LED_Off(1)
#define SYS_START LED_On(0)
#define SYS_COMPLETE LED_Off(0)

/* Run software SoftMax on unloaded data */
void softmax_q17p14_q15(const q31_t * vec_in, const uint16_t dim_vec, q15_t * p_out);
/* Shift the input, then calculate SoftMax */
void softmax_shift_q17p14_q15(q31_t * vec_in, const uint16_t dim_vec, uint8_t in_shift, q15_t * p_out);

/* Stopwatch - holds the runtime when accelerator finishes */
extern volatile uint32_t cnn_time;

/* Custom memcopy routines used for weights and data */
void memcpy32(uint32_t *dst, const uint32_t *src, int n);
void memcpy32_const(uint32_t *dst, int n);

/* Enable clocks and power to accelerator, enable interrupt */
int cnn_enable(uint32_t clock_source, uint32_t clock_divider);

/* Disable clocks and power to accelerator */
int cnn_disable(void);

/* Perform minimum accelerator initialization so it can be configured */
int cnn_init(void);

/* Configure accelerator for the given network */
int cnn_configure(void);

/* Load accelerator weights */
int cnn_load_weights(void);

/* Verify accelerator weights (debug only) */
int cnn_verify_weights(void);

/* Load accelerator bias values (if needed) */
int cnn_load_bias(void);

/* Start accelerator processing */
int cnn_start(void);

/* Force stop accelerator */
int cnn_stop(void);

/* Continue accelerator after stop */
int cnn_continue(void);

/* Unload results from accelerator */
int cnn_unload(uint32_t *out_buf);

/* Turn on the boost circuit */
int cnn_boost_enable(mxc_gpio_regs_t *port, uint32_t pin);

/* Turn off the boost circuit */
int cnn_boost_disable(mxc_gpio_regs_t *port, uint32_t pin);

#endif // __CNN_H__
