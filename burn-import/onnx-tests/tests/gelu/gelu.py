#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/gelu/gelu.onnx

import torch
import torch.nn as nn
import torch.onnx

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x)

def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "gelu.onnx"
    test_input = torch.tensor([[[[1.0, 4.0, 9.0, 25.0]]]], device=device)

    torch.onnx.export(model, (test_input), onnx_name,
                      verbose=False, 
                      opset_version=16,  
                    #   opset_version=20, TODO: uncomment this when PyTorch supports it
                    #  Note: OPSET 20 is required for GELU to be exported otherwise 
                    # op is broken down into multiple ops
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX)

    print("Finished exporting model to {}".format(onnx_name))

    # Output some test data for use in the test
    print("Test input data: {}".format(test_input))
    output = model.forward(test_input)
    print("Test output data: {}".format(output))

if __name__ == '__main__':
    main()
