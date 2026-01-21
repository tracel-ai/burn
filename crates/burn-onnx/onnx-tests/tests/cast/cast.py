#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/cast/cast.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        x_bool,
        x_int,
        x_float,
        x_scalar,
    ):
        # NOTE: we clone same-type casts for int and bool, otherwise the exporter would
        # link other type casts to the output of the bool cast, leading to additional casts
        return (
            x_bool.clone().bool(),
            x_bool.int(),
            x_bool.float(),
            x_int.bool(),
            x_int.clone().int(),
            x_int.float(),
            x_float.bool(),
            x_float.int(),
            x_float.float(),
            x_scalar.int(),
        )


def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "cast.onnx"
    test_bool = torch.ones((2, 1), device=device, dtype=torch.bool)
    test_int = torch.ones((2, 1), device=device, dtype=torch.int)
    test_float = torch.ones((2, 1), device=device, dtype=torch.float)
    test_scalar = torch.ones(1, device=device, dtype=torch.float).squeeze()
    test_input = (test_bool, test_int, test_float, test_scalar)

    # NOTE: torch exports logical_not with a cast node even if the input is already bool
    # https://github.com/pytorch/pytorch/blob/main/torch/onnx/symbolic_opset9.py#L2204-L2207
    torch.onnx.export(model, test_input, onnx_name, verbose=False, opset_version=16)

    print(f"Finished exporting model to {onnx_name}")

    # Output some test data for use in the test
    print(f"Test input data: {test_input}")
    output = model.forward(*test_input)
    print(f"Test output data: {output}")


if __name__ == "__main__":
    main()
