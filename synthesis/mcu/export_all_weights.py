"""
=============================================================================
[MCU Weights Extraction Utility: CMSIS-NN Exporter]
Description: 
Bridges the gap between PyTorch QAT output and ARM CMSIS-NN formatting.
1. Extracts quantized 8-bit weights from the model.
2. Transposes 1D Conv weights to 2D (H=1) format required by CMSIS-NN.
3. Injects dummy zero-bias arrays (since the architecture is bias=False).
4. Generates a ready-to-compile 'mcu_weights.h' C-header file.
=============================================================================
"""
import argparse
import torch
import numpy as np

def format_conv_weight(weight_tensor):
    w = weight_tensor.numpy().astype(np.int8)
    # [Data Formatting] Map 1D Conv to 2D (H=1) for CMSIS-NN compatibility
    # PyTorch: [Out, 1, In, Width] -> CMSIS-NN: [Out, 1, Width, In]
    w = np.expand_dims(w, axis=1) 
    w = np.transpose(w, (0, 1, 3, 2)) 
    return w.flatten()

def write_c_array(f, var_name, data_flat, is_int32=False):
    type_name = "int32_t" if is_int32 else "int8_t"
    f.write(f"const {type_name} {var_name}[{len(data_flat)}] = {{\n    ")
    for i, val in enumerate(data_flat):
        f.write(f"{val}, ")
        if (i + 1) % 12 == 0:
            f.write("\n    ")
    f.write("\n};\n\n")

def main(checkpoint_path, output_header):
    print("Starting custom weight extraction for CMSIS-NN...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']

    # Target layers based on ResNet-lite architecture
    layers = [
        'voice_conv1', 'voice_conv2', 'voice_conv3', 'shortcut', 'voice_conv4',
        'kws_conv1', 'kws_conv2', 'kws_conv3', 'kws_conv4'
    ]

    with open(output_header, 'w') as f:
        f.write("#ifndef MCU_WEIGHTS_H\n#define MCU_WEIGHTS_H\n\n#include <stdint.h>\n\n")

        # 1. Extract Convolutional Layers
        for name in layers:
            wt_key = f"{name}.op.weight"
            
            if wt_key in state_dict:
                wt_tensor = state_dict[wt_key]
                wt_flat = format_conv_weight(wt_tensor)
                write_c_array(f, f"{name}_wt", wt_flat)
                
                # [Hardware Alignment] Inject dummy zero-biases for bias=False layers
                out_channels = wt_tensor.shape[0]
                write_c_array(f, f"{name}_bias", np.zeros(out_channels, dtype=np.int32), is_int32=True)
            else:
                print(f"[Warning] Key not found: {wt_key}")

        # 2. Extract Fully Connected Layer
        fc_wt_key = 'fc.op.weight'
        if fc_wt_key in state_dict:
            fc_wt_tensor = state_dict[fc_wt_key]
            fc_wt_flat = fc_wt_tensor.numpy().astype(np.int8).flatten()
            write_c_array(f, "fc_wt", fc_wt_flat)
            
            num_classes = fc_wt_tensor.shape[0]
            write_c_array(f, "fc_bias", np.zeros(num_classes, dtype=np.int32), is_int32=True)
        else:
            print("[Warning] FC layer not found!")

        f.write("#endif // MCU_WEIGHTS_H\n")

    print(f"🎉 Success! '{output_header}' has been generated.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export PyTorch weights to C header.")
    parser.add_argument('--checkpoint', type=str, default='qat_best-q.pth.tar', help='Input checkpoint path')
    parser.add_argument('--output', type=str, default='mcu_weights.h', help='Output C header file name')
    args = parser.parse_args()
    main(args.checkpoint, args.output)