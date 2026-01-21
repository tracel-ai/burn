#!/usr/bin/env python3

# Used to generate model: lstm_bidirectional.onnx
# LSTM with bidirectional direction, bias enabled

import torch
import torch.nn as nn


class BiLstmModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False):
        super(BiLstmModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            bidirectional=True,
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

    print("Creating Bidirectional LSTM model...")
    model = BiLstmModel(input_size=input_size, hidden_size=hidden_size, bias=True)
    model.eval()

    device = torch.device("cpu")

    # Create test input: [seq_length, batch_size, input_size]
    # Using seq-first layout (ONNX default, layout=0)
    test_input = torch.randn(seq_length, batch_size, input_size, device=device)

    file_name = "lstm_bidirectional.onnx"

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
    print(f"Output shape: {output.shape}")  # [seq_length, batch_size, 2*hidden_size]
    print(f"h_n shape: {h_n.shape}")  # [2, batch_size, hidden_size]
    print(f"c_n shape: {c_n.shape}")  # [2, batch_size, hidden_size]

    # Print sums for verification in tests
    print(f"Output sum: {output.sum().item()}")
    print(f"h_n sum: {h_n.sum().item()}")
    print(f"c_n sum: {c_n.sum().item()}")

    # Print the test input for use in Rust tests
    print("\nTest input values:")
    print(test_input)


if __name__ == "__main__":
    main()
