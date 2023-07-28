# used to generate model: burn-import/tests/data/conv2d/conv2d.onnx

import torch
import torch.nn as nn
import onnx
from onnxoptimizer import optimize

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(16, 36, (3, 5), groups = 2, stride=(2, 1), padding=(4, 2), dilation=(3, 1))

    def forward(self, x):
        x = self.conv1(x)
        return x

def main():

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    dummy_input = torch.randn(20, 16, 50, 100, device=device)
    torch.onnx.export(model, dummy_input, "conv2d.onnx",
                      verbose=True, opset_version=16)

    # Apply the optimization pass to simplify the model
    onnx_model = onnx.load("conv2d.onnx")
    optimized_model = optimize(onnx_model)

    # Save the optimized model
    onnx.save(optimized_model, "conv2d.onnx")


if __name__ == '__main__':
    main()
