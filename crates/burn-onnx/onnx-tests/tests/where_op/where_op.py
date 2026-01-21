#!/usr/bin/env python3

# used to generate models:
#  where_op.onnx
#  where_op_broadcast.onnx
#  where_op_scalar_x.onnx
#  where_op_scalar_y.onnx

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, condition, x, y):
        return torch.where(condition, x, y)


def create_model(name: str, device: torch.device, mask: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
    print(f"--- {name} ---")
    # Export to onnx
    model = Model()
    model.eval()
    onnx_name = f"{name}.onnx"
    test_input = (mask, x, y)

    torch.onnx.export(model, (test_input), onnx_name, verbose=False, opset_version=16)

    print(f"Finished exporting model to {onnx_name}")

    # Output some test data for use in the test
    print(f"Test input data: {test_input}")
    output = model.forward(*test_input)
    print(f"Test output data: {output}")

def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)
    device = torch.device("cpu")

    mask = torch.tensor([[True, False], [False, True]], device=device)
    x = torch.ones(2, 2, device=device)
    y = torch.zeros(2, 2, device=device)
    mask_scalar = torch.tensor(True, device=device)
    x_scalar = torch.tensor(1., device=device)
    y_scalar = torch.tensor(0., device=device)
    create_model("where_op", device, mask, x, y)
    create_model("where_op_broadcast", device, mask, x[0], y[0])
    create_model("where_op_scalar_x", device, mask, x_scalar, y)
    create_model("where_op_scalar_y", device, mask, x, y_scalar)
    create_model("where_op_all_scalar", device, mask_scalar, x_scalar, y_scalar)
    


if __name__ == "__main__":
    main()