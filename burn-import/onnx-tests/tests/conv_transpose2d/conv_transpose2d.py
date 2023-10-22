#!/usr/bin/env python3

# used to generate model: transpose.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        weights = torch.tensor(
         [[[[ 1.01314831, -0.33081785,  0.51769304],
          [ 0.38777846, -0.57967722, -0.16911523],
          [ 1.93116057,  1.01186383, -0.47520983]],

         [[-0.49202651,  0.27037355, -0.56282419],
          [ 0.67934036,  0.44053012,  1.14150798],
          [ 0.01856458,  0.07325476,  0.81865305]],

         [[ 1.48047388,  0.34492949, -1.42412651],
          [-0.11632604, -0.97267389,  0.95845777],
          [-1.43352103, -0.56652677, -0.42528340]],

         [[ 0.26251873, -1.43912423,  0.52138168],
          [ 0.34875169,  0.96759415, -2.29333448],
          [ 0.49756259, -0.42572311, -1.33714700]]],


        [[[-0.19333650,  0.65264362, -1.90055323],
          [ 0.22857653, -0.98080349,  0.19473360],
          [-1.65352094,  0.68141943,  1.46111941]],

         [[-0.30975291, -1.60216033,  1.35289693],
          [ 0.57120258,  1.11791027, -1.29557133],
          [ 0.05027643, -0.58548123, -0.38999653]],

         [[ 0.03581761,  0.12058873,  0.96244991],
          [-0.33701515, -1.17533362,  0.35805708],
          [ 0.47876790,  1.35370004, -0.15933107]],

         [[-0.42494369, -0.52075714, -0.93200612],
          [ 0.18516134,  1.06869185,  1.30653441],
          [ 0.45983452,  0.26177797, -0.75993484]]]
        ])

        x = nn.functional.conv_transpose2d(x, weights, None)
        return x


def main():

    # Set seed for reproducibility
    torch.manual_seed(42)

    torch.set_printoptions(precision=8)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    file_name = "conv_transpose2d.onnx"
    test_input = torch.randn(1, 2, 3, 3, device=device)
    torch.onnx.export(model, test_input, file_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(file_name))

    # Output some test data for use in the test
    print("Test input data: {}".format(test_input))
    print("Test input data shape: {}".format(test_input.shape))
    output = model.forward(test_input)
    print("Test output data shape: {}".format(output.shape))
    print("Test output data: {}".format(output))


if __name__ == '__main__':
    main()
