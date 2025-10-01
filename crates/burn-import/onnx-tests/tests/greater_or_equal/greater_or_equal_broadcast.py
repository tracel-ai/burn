#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/greater_or_equal/greater_or_equal_broadcast.onnx

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        return torch.ge(x, y)

def main():
    # Set seed for reproducibility
    torch.manual_seed(42)
    torch.set_printoptions(precision=8)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    onnx_name = "greater_or_equal_broadcast.onnx"

    # Test broadcasting with different shapes
    # Shape [4, 1] vs [1, 4] - should broadcast to [4, 4]
    test_input1 = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device=device)
    test_input2 = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=device)
    
    torch.onnx.export(model, (test_input1, test_input2), onnx_name, 
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(onnx_name))

    print("Test input1 shape: {}, data: {}".format(test_input1.shape, test_input1))
    print("Test input2 shape: {}, data: {}".format(test_input2.shape, test_input2))
    output = model.forward(test_input1, test_input2)
    print("Test output shape: {}, data: {}".format(output.shape, output))

if __name__ == '__main__':
    main()