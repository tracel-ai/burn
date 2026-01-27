#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/shape/shape.onnx

import onnx
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, inp: torch.Tensor, shape_src: torch.Tensor) -> torch.Tensor:
        return inp.expand_as(shape_src)


def main():
    # Set seed for reproducibility
    torch.manual_seed(42)

    torch.set_printoptions(precision=8)

    # Export to onnx
    device = torch.device("cpu")
    model = Model()
    model.eval()
    test_input = torch.ones(4, 1, device=device)
    test_shape_src = torch.ones(4, 4, device=device)
    file_name = "expand_shape.onnx"

    torch.onnx.export(
        model,
        (test_input, test_shape_src),
        file_name,
        input_names=["inp", "shape_src"],
        verbose=False,
        opset_version=16,
    )

    print(f"Finished exporting model to {file_name}")

    # Output some test data for use in the test
    print(f"Test input data: {test_input}")
    print(f"Test input data shape: {test_input.shape}")
    print(f"Test shape source tensor shape: {test_input.shape}")
    output = model.forward(test_input, test_shape_src)
    print(f"Test output data shape: {output.shape}")

    print(f"Test output: {output}")


if __name__ == "__main__":
    main()
