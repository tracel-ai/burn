#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/shape/shape.onnx

import onnx
import torch
import torch.nn as nn


# Trace with TorchScript to return the shape tensor (otherwise, would gather the shape
# of each dim as a scalar)
@torch.jit.script
def shape(x):
    return torch.tensor(x.shape)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return shape(x)


def main():
    # Set seed for reproducibility
    torch.manual_seed(42)

    torch.set_printoptions(precision=8)

    # Export to onnx
    device = torch.device("cpu")
    model = Model()
    model.eval()
    test_input = torch.ones(4, 2, device=device)
    file_name = "shape.onnx"

    torch.onnx.export(
        model,
        test_input,
        file_name,
        input_names=["x"],
        dynamic_axes={"x": {0: "b"}},
        verbose=False,
        opset_version=16,
    )

    m = onnx.load(file_name)
    # Remove cast node
    m.graph.node.pop(1)
    m.graph.node[0].output[0] = m.graph.output[0].name
    onnx.save(m, file_name)

    print(f"Finished exporting model to {file_name}")

    # Output some test data for use in the test
    print(f"Test input data: {test_input}")
    print(f"Test input data shape: {test_input.shape}")
    output = model.forward(test_input)
    # print(f"Test output data shape: {output.shape}")

    print(f"Test output: {output}")


if __name__ == "__main__":
    main()
