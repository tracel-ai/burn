#!/usr/bin/env python3

import torch
import torch.nn as nn

CONST_VALUE = torch.tensor([[2, 2],
                            [2, 2]])
CONST_SHAPE = CONST_VALUE.shape

class ConstantTensorModel(nn.Module):
    def __init__(self, const_dtype: torch.dtype):
        super().__init__()
        self.const_tensor = CONST_VALUE.to(const_dtype)

    def forward(self, x):
        return self.const_tensor + x


def export_model(model: nn.Module, dummy_input: torch.Tensor, file_name: str):
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        file_name,
        verbose=False,
        opset_version=16,
        do_constant_folding=False,
    )
    print(f"Finished exporting model to {file_name}")

    # Output some test data for demonstration
    test_input = dummy_input.clone()
    print(dummy_input.dtype, "test input:", test_input)
    output = model.forward(test_input)
    print(dummy_input.dtype, "test output:", output)
    print("")


def main():
    device = torch.device("cpu")

    # Export with a float32 tensor constant
    model_f32 = ConstantTensorModel(torch.float32)
    f32_input = torch.randn(CONST_SHAPE, dtype=torch.float32, device=device)
    export_model(model_f32, f32_input, "constant_tensor_f32.onnx")

    # Export with a float64 tensor constant
    model_f64 = ConstantTensorModel(torch.float64)
    f64_input = torch.randn(CONST_SHAPE, dtype=torch.float64, device=device)
    export_model(model_f64, f64_input, "constant_tensor_f64.onnx")


if __name__ == "__main__":
    main()
