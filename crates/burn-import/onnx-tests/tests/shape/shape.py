#!/usr/bin/env python3


import torch
import torch.nn as nn
from torch.onnx import operators

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        x = self.fc(x)
        x_shape = operators.shape_as_tensor(x)
        return x, x_shape


def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Export to onnx
    model = Model()
    # model.eval()
    device = torch.device("cpu")
    onnx_name = "shape.onnx"
    test_input1 = torch.randn(1, 10)
    torch.onnx.export(model, test_input1, onnx_name,
                      verbose=False, opset_version=16,
                      input_names=["input"],
                      output_names=["output", "output_shape"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

    print("Finished exporting model to {}".format(onnx_name))


    print("Test input data: {}".format(test_input1))
    output1, output2 = model.forward(test_input1)
    print("Test output data: {}, {}".format(output1, output2))


if __name__ == '__main__':
    main()
