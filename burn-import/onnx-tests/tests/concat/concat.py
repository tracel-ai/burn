#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/concat/concat.onnx

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        # Concatenate along the channel dimension
        y = torch.cat((x,x), 1)
        x = torch.cat((x,y), 1)
        z = torch.cat((y,y), 1)
        x = torch.cat((x,y,z), 1)
        return x

def main():

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "concat.onnx"
    dummy_input = torch.randn(1,2,3,5, device=device)
    torch.onnx.export(model, dummy_input, onnx_name,
                      verbose=False, opset_version=16)
    
    print("Finished exporting model to {}".format(onnx_name))

    # Output some test data for use in the test
    test_input = torch.randn(1,2,3,5, device=device)
    print("Test input data shape: {}".format(test_input.shape))
    output = model.forward(test_input)

    print("Test output data shape: {}".format(output.shape))



if __name__ == '__main__':
    main()
