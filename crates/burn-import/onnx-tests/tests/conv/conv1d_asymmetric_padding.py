#!/usr/bin/env python3

# used to generate model: conv1d_asymmetric_padding.onnx

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnx.reference import ReferenceEvaluator

# must set for testing against crate
torch.manual_seed(0)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Create a Conv1d without padding - we'll apply asymmetric padding manually
        self.conv1 = nn.Conv1d(4, 6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        # Apply asymmetric padding: (left=1, right=2)
        # PyTorch F.pad takes (left, right) for 1D
        x = F.pad(x, (1, 2), mode='constant', value=0)
        x = self.conv1(x)
        return x


def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    file_name = "conv1d_asymmetric_padding.onnx"
    test_input = torch.ones(2, 4, 10, device=device)

    # Export with dynamo exporter (opset 18)
    torch.onnx.export(model, test_input, file_name, verbose=False, opset_version=18)

    # Load model and convert external data to embedded
    onnx_model = onnx.load(file_name, load_external_data=True)
    # Save with all data embedded
    onnx.save(onnx_model, file_name, save_as_external_data=False)

    print("Finished exporting model to {}".format(file_name))

    # Output some test data for use in the test
    print("Test input data shape of ones: {}".format(test_input.shape))
    output = model.forward(test_input)
    print("Test output data shape: {}".format(output.shape))

    # Verify with ONNX ReferenceEvaluator
    ref = ReferenceEvaluator(file_name)
    ref_output = ref.run(None, {"x": test_input.numpy()})[0]

    output_sum = output.sum().item()
    ref_sum = ref_output.sum()

    print("PyTorch output sum: {}".format(output_sum))
    print("ReferenceEvaluator output sum: {}".format(ref_sum))


if __name__ == "__main__":
    main()
