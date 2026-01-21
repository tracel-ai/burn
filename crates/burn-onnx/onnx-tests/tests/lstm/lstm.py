#!/usr/bin/env python3

# Used to generate model: lstm.onnx
# LSTM with forward direction, bias enabled

import torch
import torch.nn as nn


class LstmModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False):
        super(LstmModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            bidirectional=False,
        )

    def forward(self, x):
        # Returns (output, (h_n, c_n))
        output, (h_n, c_n) = self.lstm(x)
        return output, h_n, c_n


def main():
    torch.manual_seed(42)

    # Model parameters
    input_size = 4
    hidden_size = 8
    seq_length = 5
    batch_size = 2

    print("Creating LSTM model...")
    model = LstmModel(input_size=input_size, hidden_size=hidden_size, bias=True)
    model.eval()

    device = torch.device("cpu")

    # Create test input: [seq_length, batch_size, input_size]
    # Using seq-first layout (ONNX default, layout=0)
    test_input = torch.randn(seq_length, batch_size, input_size, device=device)

    file_name = "lstm.onnx"

    # Export to ONNX
    torch.onnx.export(
        model,
        test_input,
        file_name,
        verbose=False,
        opset_version=16,
        input_names=["input"],
        output_names=["output", "h_n", "c_n"],
        dynamic_axes=None,  # Static shapes for simpler testing
    )

    print(f"Finished exporting model to {file_name}")

    # Run inference to get expected outputs
    with torch.no_grad():
        output, h_n, c_n = model(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"h_n shape: {h_n.shape}")
    print(f"c_n shape: {c_n.shape}")

    # Print sums for verification in tests
    print(f"Output sum: {output.sum().item()}")
    print(f"h_n sum: {h_n.sum().item()}")
    print(f"c_n sum: {c_n.sum().item()}")


if __name__ == "__main__":
    main()
