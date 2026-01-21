#!/usr/bin/env python3

# used to generate model: random_normal_like.onnx

import torch
import torch.nn as nn


class RandomNormalLikeModel(nn.Module):
    def __init__(self):
        super(RandomNormalLikeModel, self).__init__()

    def forward(self, x):
        return torch.randn_like(x)


def main():
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Set print options for better precision output
    torch.set_printoptions(precision=8)
                           
    # Export Random NormalLike Model
    model = RandomNormalLikeModel()
    model.eval()
    device = torch.device("cpu")

    # Generate test input: a 2D matrix or batch of 2D matrices
    file_name = "random_normal_like.onnx"
    test_input = torch.randn(2, 4, 4, device=device)  # 2 batches of 4x4 matrices
    torch.onnx.export(model,
                      test_input,
                      file_name,
                      verbose=False,
                      opset_version=16)

    print("Finished exporting model to {}".format(file_name))

    # Output some test data for use in the test
    print("Test input data: {}".format(test_input))
    print("Test input data shape: {}".format(test_input.shape))
    output = model.forward(test_input)
    print("Test output data shape: {}".format(output.shape))
    print("Test output: {}".format(output))


if __name__ == '__main__':
    main()