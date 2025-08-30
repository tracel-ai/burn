#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/less_or_equal/less_or_equal_broadcast.onnx

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        return torch.le(x, y)

def main():
    # Set seed for reproducibility
    torch.manual_seed(42)
    torch.set_printoptions(precision=8)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    onnx_name = "less_or_equal_broadcast.onnx"

    # Test broadcasting with scalar-like shapes
    # Shape [1] vs [4, 4] - should broadcast to [4, 4]
    test_input1 = torch.tensor([2.5], device=device)
    test_input2 = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                                 [2.0, 2.5, 3.0, 3.5],
                                 [1.5, 2.5, 3.5, 4.5],
                                 [0.5, 1.5, 2.5, 3.5]], device=device)
    
    torch.onnx.export(model, (test_input1, test_input2), onnx_name, 
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(onnx_name))

    print("Test input1 shape: {}, data: {}".format(test_input1.shape, test_input1))
    print("Test input2 shape: {}, data: {}".format(test_input2.shape, test_input2))
    output = model.forward(test_input1, test_input2)
    print("Test output shape: {}, data: {}".format(output.shape, output))

if __name__ == '__main__':
    main()