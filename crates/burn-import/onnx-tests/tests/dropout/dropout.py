#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/dropout/dropout.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.dropout(x)
        return x


def main():

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    file_name = "dropout.onnx"
    test_input = torch.ones(2, 4, 10, 15, device=device)
    torch.onnx.export(model, test_input, file_name,
                      training=torch.onnx.TrainingMode.TRAINING,
                      do_constant_folding=False,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(file_name))

    # Output some test data for use in the test
    print("Test input data shape of ones: {}".format(test_input.shape))
    output = model.forward(test_input)
    print("Test output data shape: {}".format(output.shape))

    sum = output.sum().item()

    print("Test output sum: {}".format(sum))


if __name__ == '__main__':
    main()
