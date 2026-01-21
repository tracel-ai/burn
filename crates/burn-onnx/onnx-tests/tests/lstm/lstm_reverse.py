#!/usr/bin/env python3

# Used to generate model: lstm_reverse.onnx
# LSTM with reverse direction, bias enabled
#
# PyTorch doesn't have a native reverse-only LSTM, so we:
# 1. Export a forward LSTM from PyTorch
# 2. Modify the ONNX model to set direction="reverse"
# 3. Compute expected values by manually reversing input/output

import torch
import torch.nn as nn
import onnx
import numpy as np
from onnx import helper, numpy_helper


class ForwardLstmModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False):
        super(ForwardLstmModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            bidirectional=False,
        )

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return output, h_n, c_n


def main():
    torch.manual_seed(42)

    # Model parameters
    input_size = 4
    hidden_size = 8
    seq_length = 5
    batch_size = 2

    print("Creating Forward LSTM model (will be converted to reverse)...")
    model = ForwardLstmModel(input_size=input_size, hidden_size=hidden_size, bias=True)
    model.eval()

    device = torch.device("cpu")

    # Create test input: [seq_length, batch_size, input_size]
    test_input = torch.randn(seq_length, batch_size, input_size, device=device)

    temp_file = "lstm_forward_temp.onnx"
    file_name = "lstm_reverse.onnx"

    # Export forward LSTM to ONNX first
    torch.onnx.export(
        model,
        test_input,
        temp_file,
        verbose=False,
        opset_version=16,
        input_names=["input"],
        output_names=["output", "h_n", "c_n"],
        dynamic_axes=None,
    )

    # Load and modify the model to have direction="reverse"
    onnx_model = onnx.load(temp_file)

    # Find the LSTM node and change direction attribute
    for node in onnx_model.graph.node:
        if node.op_type == "LSTM":
            # Remove existing direction attribute if any
            for i, attr in enumerate(node.attribute):
                if attr.name == "direction":
                    del node.attribute[i]
                    break
            # Add direction="reverse"
            node.attribute.append(helper.make_attribute("direction", "reverse"))

    # Save the modified model
    onnx.save(onnx_model, file_name)

    # Clean up temp file
    import os
    os.remove(temp_file)

    print(f"Finished exporting model to {file_name}")

    # Compute expected values for reverse LSTM
    # For reverse direction:
    # - Process input from seq_length-1 to 0
    # - Y[t] contains result after processing input[t:seq_length-1] in reverse
    # We simulate this by flipping input, running forward, then flipping output
    with torch.no_grad():
        input_reversed = torch.flip(test_input, dims=[0])
        output_fwd, h_n, c_n = model(input_reversed)
        output = torch.flip(output_fwd, dims=[0])

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"h_n shape: {h_n.shape}")
    print(f"c_n shape: {c_n.shape}")

    # Print sums for verification in tests
    print(f"Output sum: {output.sum().item()}")
    print(f"h_n sum: {h_n.sum().item()}")
    print(f"c_n sum: {c_n.sum().item()}")

    # Print the test input for use in Rust tests
    print("\nTest input values:")
    print(test_input)


if __name__ == "__main__":
    main()
