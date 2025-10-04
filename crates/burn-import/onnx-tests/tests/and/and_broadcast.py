#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/and/and_broadcast.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x_3d, y_2d, a_2d, b_3d):
        # Case 1: 3D tensor & 2D tensor (broadcast 2D to 3D)
        result1 = x_3d & y_2d
        
        # Case 2: 2D tensor & 3D tensor (broadcast 2D to 3D)
        result2 = a_2d & b_3d
        
        return result1, result2


def main():
    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "and_broadcast.onnx"
    
    # Create dummy inputs with different ranks
    dummy_x_3d = (torch.rand(2, 3, 4, device=device) < 0.5).bool()  # 3D tensor
    dummy_y_2d = (torch.rand(3, 4, device=device) < 0.5).bool()     # 2D tensor
    dummy_a_2d = (torch.rand(3, 4, device=device) < 0.5).bool()     # 2D tensor
    dummy_b_3d = (torch.rand(2, 3, 4, device=device) < 0.5).bool()  # 3D tensor

    torch.onnx.export(model, (dummy_x_3d, dummy_y_2d, dummy_a_2d, dummy_b_3d), onnx_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(onnx_name))

    # Output some test data for use in the test
    test_x_3d = torch.tensor([[[ True, False, False, False],
                               [ True, False,  True, False],
                               [False,  True,  True,  True]],

                              [[ True, False, False,  True],
                               [False, False,  True,  True],
                               [ True,  True, False,  True]]])
    
    test_y_2d = torch.tensor([[False, False,  True,  True],
                              [False,  True,  True, False],
                              [False, False, False,  True]])
    
    test_a_2d = torch.tensor([[False,  True, False, False],
                              [False, False, False,  True],
                              [ True, False, False,  True]])
    
    test_b_3d = torch.tensor([[[ True, False,  True,  True],
                               [ True,  True, False,  True],
                               [False, False, False, False]],

                              [[ True,  True,  True, False],
                               [False,  True, False, False],
                               [False, False,  True, False]]])

    print("Test input x_3d shape: {}".format(test_x_3d.shape))
    print("Test input y_2d shape: {}".format(test_y_2d.shape))
    print("Test input a_2d shape: {}".format(test_a_2d.shape))
    print("Test input b_3d shape: {}".format(test_b_3d.shape))
    
    # Run the model to get actual outputs
    with torch.no_grad():
        result1, result2 = model.forward(test_x_3d, test_y_2d, test_a_2d, test_b_3d)
    
    print("\nTest outputs:")
    print("result1 (x_3d & y_2d):")
    print(result1)
    print("\nresult2 (a_2d & b_3d):")
    print(result2)


if __name__ == '__main__':
    main()
