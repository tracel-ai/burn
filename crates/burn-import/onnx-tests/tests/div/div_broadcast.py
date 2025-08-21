#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/div/div_broadcast.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x_3d, y_2d, a_2d, b_3d):
        # Case 1: 3D tensor / 2D tensor (broadcast 2D to 3D)
        result1 = x_3d / y_2d
        
        # Case 2: 2D tensor / 3D tensor (broadcast 2D to 3D)
        result2 = a_2d / b_3d
        
        return result1, result2


def main():
    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "div_broadcast.onnx"
    
    # Create dummy inputs with different ranks
    dummy_x_3d = torch.randn(2, 3, 4, device=device)  # 3D tensor
    dummy_y_2d = torch.randn(3, 4, device=device)     # 2D tensor
    dummy_a_2d = torch.randn(3, 4, device=device)     # 2D tensor
    dummy_b_3d = torch.randn(2, 3, 4, device=device)  # 3D tensor

    torch.onnx.export(model, (dummy_x_3d, dummy_y_2d, dummy_a_2d, dummy_b_3d), onnx_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(onnx_name))

    # Output some test data for use in the test
    test_x_3d = torch.tensor([[[10.0, 20.0, 30.0, 40.0],
                                [50.0, 60.0, 70.0, 80.0],
                                [90.0, 100.0, 110.0, 120.0]],
                               [[12.0, 24.0, 36.0, 48.0],
                                [60.0, 72.0, 84.0, 96.0],
                                [108.0, 120.0, 132.0, 144.0]]])
    
    test_y_2d = torch.tensor([[2.0, 4.0, 6.0, 8.0],
                               [10.0, 12.0, 14.0, 16.0],
                               [18.0, 20.0, 22.0, 24.0]])
    
    test_a_2d = torch.tensor([[100.0, 200.0, 300.0, 400.0],
                               [500.0, 600.0, 700.0, 800.0],
                               [900.0, 1000.0, 1100.0, 1200.0]])
    
    test_b_3d = torch.tensor([[[10.0, 20.0, 30.0, 40.0],
                                [50.0, 60.0, 70.0, 80.0],
                                [90.0, 100.0, 110.0, 120.0]],
                               [[5.0, 10.0, 15.0, 20.0],
                                [25.0, 30.0, 35.0, 40.0],
                                [45.0, 50.0, 55.0, 60.0]]])

    print("Test input x_3d shape: {}".format(test_x_3d.shape))
    print("Test input y_2d shape: {}".format(test_y_2d.shape))
    print("Test input a_2d shape: {}".format(test_a_2d.shape))
    print("Test input b_3d shape: {}".format(test_b_3d.shape))
    
    # Run the model to get actual outputs
    with torch.no_grad():
        result1, result2 = model.forward(test_x_3d, test_y_2d, test_a_2d, test_b_3d)
    
    print("\nTest outputs:")
    print("result1 (x_3d / y_2d):")
    print(result1)
    print("\nresult2 (a_2d / b_3d):")
    print(result2)


if __name__ == '__main__':
    main()