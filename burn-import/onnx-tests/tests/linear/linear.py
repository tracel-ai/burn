#!/usr/bin/env python3

# used to generate model: linear.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Case 1: When input's 2d with [1, n] shape and bias is True.
        # This will produce a single Gemm ONNX node with alpha=1 and beta=1 and transB=1 attributes.
        # with 3 inputs and 1 output. The 3 inputs are the input, weight, and bias.
        self.linear1_with_gemm = nn.Linear(3, 4, bias=True)

        # Case 2: When input >= 2D but linear does not have bias.
        self.linear2_with_matmul = nn.Linear(5, 6, bias=False)

        # Case 3: When input > 2D and linear does have bias or does not have bias (doesn't matter).
        self.linear3_with_matmul = nn.Linear(7, 8, bias=True)

    def forward(self, x1, x2, x3):
        y1 = self.linear1_with_gemm(x1)
        y2 = self.linear2_with_matmul(x2)
        y3 = self.linear3_with_matmul(x3)
        return y1, y2, y3


def main():

    # Set random seed for reproducibility
    torch.manual_seed(0)

    print("Making model")
    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    print("Made model")

    file_name = "linear.onnx"
    input1 = torch.full((4, 3), 3.14, device=device)
    input2 = torch.full((2, 5), 3.14, device=device)
    input3 = torch.full((3, 2, 7), 3.14, device=device)
    torch.onnx.export(model, (input1, input2, input3), file_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(file_name))

    # Output some test data for use in the test
    print("Test input1 data shape: {}".format(input1.shape))
    print("Test input2 data shape: {}".format(input2.shape))
    print("Test input3 data shape: {}".format(input3.shape))

    output1, output2, output3 = model.forward(input1, input2, input3)

    print("Test output1 data shape: {}".format(output1.shape))
    print("Test output2 data shape: {}".format(output2.shape))
    print("Test output3 data shape: {}".format(output3.shape))

    sum1 = output1.sum().item()
    sum2 = output2.sum().item()
    sum3 = output3.sum().item()

    print("Test output sum1: {}".format(sum1))
    print("Test output sum2: {}".format(sum2))
    print("Test output sum3: {}".format(sum3))


if __name__ == "__main__":
    main()
