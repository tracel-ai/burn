#!/usr/bin/env python3

import torch
import torch.nn as nn

CONST_VALUE = 2
CONST_BOOL_VALUE = True


class ConstantModel(nn.Module):
    def __init__(self, const_dtype: torch.dtype):
        super().__init__()
        if const_dtype == torch.bool:
            self.const = torch.tensor(CONST_BOOL_VALUE).to(const_dtype)
        else:
            self.const = torch.tensor(CONST_VALUE).to(const_dtype)

    def forward(self, x):
        if x.dtype == torch.bool:
            return x | self.const  # Use logical OR for boolean tensors
        else:
            return x + self.const


def export_model(model: ConstantModel, dummy_input: torch.Tensor, file_name: str):
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
    shape = (2, 3, 4)

    model_f32 = ConstantModel(torch.float32)
    f32_input = torch.randn(shape, dtype=torch.float32, device=device)
    export_model(model_f32, f32_input, "constant_f32.onnx")

    model_f64 = ConstantModel(torch.float64)
    f64_input = torch.randn(shape, dtype=torch.float64, device=device)
    export_model(model_f64, f64_input, "constant_f64.onnx")

    model_i32 = ConstantModel(torch.int32)
    i32_input = torch.randint(
        low=-10, high=10, size=shape, device=device, dtype=torch.int32
    )
    export_model(model_i32, i32_input, "constant_i32.onnx")

    model_i64 = ConstantModel(torch.int64)
    i64_input = torch.randint(
        low=-10, high=10, size=shape, device=device, dtype=torch.int64
    )
    export_model(model_i64, i64_input, "constant_i64.onnx")

    model_bool = ConstantModel(torch.bool)
    bool_input = torch.randint(
        low=0, high=2, size=[], device=device, dtype=torch.bool
    )
    export_model(model_bool, bool_input, "constant_bool.onnx")

if __name__ == "__main__":
    main()
