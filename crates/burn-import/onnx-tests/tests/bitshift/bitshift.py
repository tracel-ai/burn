import torch
import onnx

def export_bitshift(dir: str = "LEFT"):
    class BitShiftModel(torch.nn.Module):
        def __init__(self, direction):
            super(BitShiftModel, self).__init__()
            self.direction = direction

        def forward(self, x, shift):
            if self.direction == "LEFT":
                return torch.bitwise_left_shift(x, shift)
            elif self.direction == "RIGHT":
                return torch.bitwise_right_shift(x, shift)
            else:
                raise ValueError("Invalid direction. Use 'LEFT' or 'RIGHT'.")

    model = BitShiftModel(direction=dir)  # Change to "RIGHT" to test right shift
    x = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    shift = torch.tensor([1, 1, 1, 1], dtype=torch.int32)  # Example shift tensor
    torch.onnx.export(
        model,
        (x, shift),
        f"bitshift_{dir.lower()}.onnx",
        opset_version=18,
        input_names=["x", "shift"],
        output_names=["output"],
    )

    # Scalar version
    shift_scalar = 1  # Scalar shift value
    torch.onnx.export(
        model,
        (x, shift_scalar),
        f"bitshift_{dir.lower()}_scalar.onnx",
        opset_version=18,
        input_names=["x", "shift"],
        output_names=["output"],
    )

if __name__ == "__main__":
    for direction in ["LEFT", "RIGHT"]:
        export_bitshift(direction)