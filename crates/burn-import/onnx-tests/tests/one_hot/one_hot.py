#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/one_hot/one_hot.onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class OneHotModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes  # Number of categories for one-hot encoding

    def forward(self, x):
        one_hot = F.one_hot(x, num_classes=self.num_classes)
        return one_hot  # Convert to float for compatibility

# Create model instance
num_classes = 3
model = OneHotModel(num_classes=num_classes)
model.eval()

# Example input: Tensor of class indices
test_input = torch.tensor([0, 1, 2], dtype=torch.int64)

# Export to ONNX
onnx_file = "one_hot.onnx"
torch.onnx.export(
    model,
    test_input,
    onnx_file,
    opset_version=16,
    input_names=["input"],
    output_names=["one_hot_output"],
    dynamic_axes={"input": {0: "batch_size"}, "one_hot_output": {0: "batch_size"}}
)

print(f"Finished exporting model to {onnx_file}")
print(f"Test input data of ones: {test_input}")
print(f"Test input data shape of ones: {test_input.shape}")
output = model.forward(test_input)
print(f"Test output data shape: {output.shape}")
print(f"Test output: {output}")
