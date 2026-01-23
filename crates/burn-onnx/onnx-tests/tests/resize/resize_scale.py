#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.onnx

class InterpolateModel(nn.Module):
    def __init__(self, scale_factor=None, size=None, mode='nearest', align_corners=None):
        super(InterpolateModel, self).__init__()
        self.scale_factor = scale_factor
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, size=self.size,
                                         mode=self.mode, align_corners=self.align_corners)

def export_interpolate_onnx(filename, batch_size=1, channels=1, height=6, width=6,
                            scale_factor=None, size=None, mode='nearest', rank=2, align_corners=None):
    model = InterpolateModel(scale_factor, size, mode, align_corners)
    model.eval()

    # Add seed for reproducibility
    torch.manual_seed(0)

    # Create a dummy input
    if rank == 1:
        dummy_input = torch.randn(batch_size, channels, width)
    elif rank == 2:
        dummy_input = torch.randn(batch_size, channels, height, width)
    else:
        raise ValueError("Unsupported rank. Use 1 for temporal or 2 for spatial.")

    # Export the model
    torch.onnx.export(model, dummy_input, filename,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}},
                      opset_version=17)

    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Print sum data
    print(f"Input sum: {dummy_input.sum()}")
    print(f"Output sum: {output.sum()}")

    print(f"Input: {dummy_input}")
    print(f"Output: {output}")

    print(f"Model exported to {filename}")

    print()

# Usage examples:
if __name__ == "__main__":


    # 1D (temporal) examples
    export_interpolate_onnx("resize_1d_nearest_scale.onnx", scale_factor=1.5, mode='nearest', rank=1)
    export_interpolate_onnx("resize_1d_linear_scale.onnx", scale_factor=1.5, mode='linear', rank=1, align_corners=True)

    # Cubic interpolation is not supported for 1D tensors
    # export_interpolate_onnx("resize_1d_cubic_scale.onnx", scale_factor=1.5, mode='cubic', rank=1)

    # 2D (spatial) examples
    export_interpolate_onnx("resize_2d_nearest_scale.onnx", scale_factor=1.5, mode='nearest', rank=2)
    export_interpolate_onnx("resize_2d_bilinear_scale.onnx", scale_factor=1.5, mode='bilinear', rank=2, align_corners=True)
    export_interpolate_onnx("resize_2d_bicubic_scale.onnx", scale_factor=1.5, mode='bicubic', rank=2, align_corners=True)
