#!/usr/bin/env python3

# used to generate models: squeeze_opset13.onnx,
# squeeze_opset16.onnx, and squeeze_multiple.onnx

import torch
import onnx
import torch.nn as nn
from onnx import helper, TensorProto

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.dims = 2

    def forward(self, x):
        x = torch.squeeze(x, self.dims)
        return x


def main():
    # Set seed for reproducibility
    torch.manual_seed(42)

    torch.set_printoptions(precision=8)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    test_input = torch.randn(3, 4, 1, 5, device=device)

    # Export to ONNX
    torch.onnx.export(model, test_input, "squeeze.onnx", verbose=False, opset_version=16)

    print("Finished exporting model")

    # Output some test data for use in the test
    output = model(test_input)
    print(f"Test input data: {test_input}")
    print(f"Test input data shape: {test_input.shape}")
    print(f"Test output data shape: {output.shape}")
    print(f"Test output: {output}")

    # Test for squeezing multiple dimensions
    test_input_ms = helper.make_tensor_value_info("input", TensorProto.FLOAT, [3, 4, 1, 5, 1])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 4, 5])
    squeeze = helper.make_node(op_type="Squeeze", inputs=["input", "axes"], outputs=["output"], name="SqueezeOp")
    axes = helper.make_tensor("axes", TensorProto.INT64, dims=[2], vals=[2, 4])
    graph = helper.make_graph([squeeze], "SqueezeMultiple", [test_input_ms], [output], [axes])
    opset = helper.make_opsetid("", 16)
    m = helper.make_model(graph, opset_imports=[opset])

    onnx.checker.check_model(m, full_check=True)
    onnx.save(m, "squeeze_multiple.onnx")

    print("Finished exporting model with multiple squeeze axes specified to 13")

if __name__ == "__main__":
    main()
