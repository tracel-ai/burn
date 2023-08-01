# used to generate model: onnx-tests/tests/add/add.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):

        # Declare a constant tensor
        a = torch.ones(1, 1, 1, 4)

        # Add a tensor and a constant
        x = x + a

        # Add a tensor and a scalar
        x = x + 5

        return x


def main():

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "add.onnx"
    dummy_input = torch.randn(1, 2, 3, 4, device=device)

    torch.onnx.export(model, dummy_input, onnx_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(onnx_name))

    # Output some test data for use in the test
    test_input = torch.tensor([[[[1, 2, 3, 4]]]], dtype=torch.float32)
    print("Test input data: {}".format(test_input))
    output = model.forward(test_input)
    print("Test output data: {}".format(output))


if __name__ == '__main__':
    main()
