#!/usr/bin/env python3
# used to generate model: onnx-tests/tests/bitwise_and/bitwise_and.onnx

import torch
import onnx

def export_bitwise_and():
    class BitwiseAndModel(torch.nn.Module):
        def forward(self, x, y):
            if isinstance(y, int):
                # If y is a scalar, convert it to a tensor
                y = torch.tensor([y], dtype=x.dtype)
            return torch.bitwise_and(x, y)

    model = BitwiseAndModel()
    x = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    y = torch.tensor([4, 3, 2, 1], dtype=torch.int32)
    torch.onnx.export(
        model,
        (x, y),
        "bitwise_and.onnx",
        opset_version=18,
        input_names=["x", "y"],
        output_names=["output"],
    )

    # Scalar version
    and_scalar = 2  # Scalar shift value
    torch.onnx.export(
        model,
        (x, and_scalar),
        f"bitwise_and_scalar.onnx",
        opset_version=18,
        input_names=["x", "y"],
        output_names=["output"],
    )

if __name__ == "__main__":
    export_bitwise_and()