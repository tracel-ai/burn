#!/usr/bin/env python3

# used to generate model: one_hot.onnx

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
   def __init__(self):
       super(Model, self).__init__()

   def forward(self, x):
       x = F.one_hot(x, num_classes=3)
       return x

def main():
    torch.manual_seed(42)

    torch.set_printoptions(precision=8)

    model = Model()
    model.eval()
    device = torch.device("cpu")


    file_name = "one_hot.onnx"
    test_input = torch.tensor([1, 0, 2], device=device)
    torch.onnx.export(model, test_input, file_name,
                      verbose=False, opset_version=16)
    print("Finished exporting model to {}".format(file_name))
    print("Test input data of ones: {}".format(test_input))
    print("Test input data shape of ones: {}".format(test_input.shape))
    output = model.forward(test_input)
    print("Test output data shape: {}".format(output.shape))
    print("Test output: {}".format(output))

if __name__ == '__main__':
    main()
