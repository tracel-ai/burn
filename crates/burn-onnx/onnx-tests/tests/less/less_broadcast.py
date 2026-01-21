#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/less/less_broadcast.onnx

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        return torch.lt(x, y)

def main():
    # Set seed for reproducibility
    torch.manual_seed(42)
    torch.set_printoptions(precision=8)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    onnx_name = "less_broadcast.onnx"

    # Test broadcasting with different shapes
    # Shape [1, 77] vs [77, 1] - this is the pattern from CLIP that was failing
    test_input1 = torch.randn(1, 77, device=device)
    test_input2 = torch.randn(77, 1, device=device)
    
    torch.onnx.export(model, (test_input1, test_input2), onnx_name, 
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(onnx_name))

    print("Test input1 shape: {}".format(test_input1.shape))
    print("Test input2 shape: {}".format(test_input2.shape))
    output = model.forward(test_input1, test_input2)
    print("Test output shape: {}".format(output.shape))
    # Just print a sample of the output since it's large
    print("Test output sample (first 5x5): {}".format(output[:5, :5]))

if __name__ == '__main__':
    main()