#!/usr/bin/env python3

# Used to generate model: lstm_with_initial_state.onnx
# LSTM with forward direction, bias enabled, and initial states (h_0, c_0)

import torch
import torch.nn as nn


class LstmWithInitialStateModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False):
        super(LstmWithInitialStateModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            bidirectional=False,
        )

    def forward(self, x, h_0, c_0):
        # h_0 and c_0 are initial hidden and cell states
        # Shape: [num_layers * num_directions, batch_size, hidden_size]
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        return output, h_n, c_n


def main():
    torch.manual_seed(42)

    # Model parameters
    input_size = 4
    hidden_size = 8
    seq_length = 5
    batch_size = 2

    print("Creating LSTM model with initial states...")
    model = LstmWithInitialStateModel(input_size=input_size, hidden_size=hidden_size, bias=True)
    model.eval()

    device = torch.device("cpu")

    # Create test input: [seq_length, batch_size, input_size]
    test_input = torch.randn(seq_length, batch_size, input_size, device=device)

    # Create initial states: [num_layers * num_directions, batch_size, hidden_size]
    # For single-layer unidirectional: [1, batch_size, hidden_size]
    h_0 = torch.randn(1, batch_size, hidden_size, device=device)
    c_0 = torch.randn(1, batch_size, hidden_size, device=device)

    file_name = "lstm_with_initial_state.onnx"

    # Export to ONNX
    torch.onnx.export(
        model,
        (test_input, h_0, c_0),
        file_name,
        verbose=False,
        opset_version=16,
        input_names=["input", "h_0", "c_0"],
        output_names=["output", "h_n", "c_n"],
        dynamic_axes=None,
    )

    print(f"Finished exporting model to {file_name}")

    # Run inference to get expected outputs
    with torch.no_grad():
        output, h_n, c_n = model(test_input, h_0, c_0)

    print(f"Input shape: {test_input.shape}")
    print(f"h_0 shape: {h_0.shape}")
    print(f"c_0 shape: {c_0.shape}")
    print(f"Output shape: {output.shape}")
    print(f"h_n shape: {h_n.shape}")
    print(f"c_n shape: {c_n.shape}")

    # Print sums for verification in tests
    print(f"\nOutput sum: {output.sum().item()}")
    print(f"h_n sum: {h_n.sum().item()}")
    print(f"c_n sum: {c_n.sum().item()}")

    # Print the test input values for use in Rust tests
    print("\nTest input values:")
    print(test_input)

    print("\nh_0 values:")
    print(h_0)

    print("\nc_0 values:")
    print(c_0)


if __name__ == "__main__":
    main()
