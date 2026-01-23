#!/usr/bin/env python3
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, fill_value_float, fill_value_int, fill_value_bool):
        super(Model, self).__init__()
        self.fill_value_float = fill_value_float
        self.fill_value_int = fill_value_int
        self.fill_value_bool = fill_value_bool

    def forward(self, x):
        # Use full_like, which will be exported as ConstantOfShape
        f = torch.full_like(x, self.fill_value_float, dtype=torch.float)
        i = torch.full_like(x, self.fill_value_int, dtype=torch.int)
        # Convert bool to int (1 or 0) for compatibility
        b = torch.full_like(x, int(self.fill_value_bool), dtype=torch.bool)
        return f, i, b

def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Create an instance of the model
    model = Model(3.0, 5, True)

    # Create a dummy input
    test_input = torch.randn(2, 3, 4)

    file_name = "constant_of_shape_full_like.onnx"

    # Export the model to ONNX
    torch.onnx.export(model, test_input, file_name,
                      verbose=False, opset_version=16,
                      input_names=['input'],
                      output_names=['output_float', 'output_int', 'output_bool'],
                      dynamic_axes={'input': {0: 'batch_size', 1: 'height', 2: 'width'},
                                    'output_float': {0: 'batch_size', 1: 'height', 2: 'width'},
                                    'output_int': {0: 'batch_size', 1: 'height', 2: 'width'},
                                    'output_bool': {0: 'batch_size', 1: 'height', 2: 'width'}})

    print(f"Finished exporting model to {file_name}")

    # Output some test data for use in the test
    print(f"Test input data shape: {test_input.shape}")
    f, i, b = model.forward(test_input)
    print(f"Test output data shape of float: {f.shape}")
    print(f"Test output data shape of int: {i.shape}")
    print(f"Test output data shape of bool: {b.shape}")

    sum_f = f.sum().item()
    sum_i = i.sum().item()
    all_b = b.all().item()
    print(f"Test output sum of float: {sum_f}")
    print(f"Test output sum of int: {sum_i}")
    print(f"Test output all of bool: {all_b}")

if __name__ == "__main__":
    main()
