#!/usr/bin/env python3

# used to generate models: scatter_pytorch.onnx (via PyTorch export, uses ScatterElements)
#                          scatter_onnx.onnx (via manual ONNX construction, uses Scatter operator)

import torch
from torch import Tensor
import torch.nn as nn
import onnx


class ScatterModel(nn.Module):
    def __init__(self, dim, index, src):
        super(ScatterModel, self).__init__()
        self.dim = dim
        self.index = index
        self.src = src

    def forward(self, x: Tensor):
        y = x.scatter(self.dim, self.index, self.src)
        return y


def build_scatter_onnx_model():
    """
    Build an ONNX model manually using the Scatter operator.
    Note: Scatter was deprecated in opset 11 in favor of ScatterElements,
    but we include it here for backwards compatibility testing.
    
    This model performs: output[i][indices[i][j]] = updates[i][j] along axis 1
    Input data shape: [3, 5]
    Indices shape: [3, 3]  (must match data shape on non-scatter dimensions for Burn compatibility)
    Updates shape: [3, 3]  (must match indices shape)
    Output shape: [3, 5]
    """
    return onnx.helper.make_model(
        ir_version=8,
        opset_imports=[onnx.helper.make_operatorsetid("", 9)],  # Scatter available in opset 9-10
        graph=onnx.helper.make_graph(
            name="scatter_graph",
            nodes=[
                onnx.helper.make_node(
                    "Scatter",
                    inputs=["data", "indices", "updates"],
                    outputs=["output"],
                    name="/Scatter",
                    axis=1
                ),
            ],
            inputs=[
                onnx.helper.make_value_info(
                    name="data",
                    type_proto=onnx.helper.make_tensor_type_proto(
                        elem_type=onnx.TensorProto.FLOAT, shape=[3, 5]
                    ),
                ),
                onnx.helper.make_value_info(
                    name="indices",
                    type_proto=onnx.helper.make_tensor_type_proto(
                        elem_type=onnx.TensorProto.INT64, shape=[3, 3]
                    ),
                ),
                onnx.helper.make_value_info(
                    name="updates",
                    type_proto=onnx.helper.make_tensor_type_proto(
                        elem_type=onnx.TensorProto.FLOAT, shape=[3, 3]
                    ),
                ),
            ],
            outputs=[
                onnx.helper.make_value_info(
                    name="output",
                    type_proto=onnx.helper.make_tensor_type_proto(
                        elem_type=onnx.TensorProto.FLOAT, shape=[3, 5]
                    ),
                )
            ]
        ),
    )


def main():
    # Set seed for reproducibility
    torch.manual_seed(42)

    torch.set_printoptions(precision=8)

    # ============================================================
    # Part 1: PyTorch export (creates ScatterElements in ONNX)
    # ============================================================
    
    # Set up the scatter parameters
    src = torch.tensor([[1., 2., 3.]])
    index = torch.tensor([[0, 1, 2]])
    dim = 1

    # Create model
    model = ScatterModel(dim=dim, index=index, src=src)
    model.eval()
    device = torch.device("cpu")

    # Create test input
    test_input = torch.zeros(3, 5, dtype=src.dtype, device=device)

    # Export to ONNX (this will use ScatterElements operator)
    torch.onnx.export(model, test_input, "scatter_pytorch.onnx", verbose=False, opset_version=16)

    print("Finished exporting PyTorch model to scatter_pytorch.onnx (uses ScatterElements)")

    # Output test data for use in the test
    output = model(test_input)
    print(f"Test input data: {test_input}")
    print(f"Test input data shape: {test_input.shape}")
    print(f"Test output data: {output}")
    print(f"Test output data shape: {output.shape}")

    # ============================================================
    # Part 2: Manual ONNX construction (uses Scatter operator)
    # ============================================================
    
    print("\n" + "="*60)
    print("Building manual ONNX model with Scatter operator...")
    
    onnx_model = build_scatter_onnx_model()
    
    # Ensure valid ONNX
    onnx.checker.check_model(onnx_model)
    
    onnx.save(onnx_model, "scatter_onnx.onnx")
    
    print("Finished exporting manual ONNX model to scatter_onnx.onnx (uses Scatter)")
    print("Scatter ONNX model inputs: data[3,5], indices[3,3], updates[3,3]")
    print("Scatter ONNX model output: output[3,5]")


if __name__ == "__main__":
    main()

