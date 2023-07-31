import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnxoptimizer import optimize


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.norm1 = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(8*6*6, 10)
        self.norm2 = nn.BatchNorm1d(10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.norm1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.norm2(x)
        output = F.log_softmax(x, dim=1)
        return output


def main():

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    dummy_input = torch.randn(1, 1, 8, 8, device=device)
    torch.onnx.export(model, dummy_input, "model1.onnx",
                      verbose=False, opset_version=16)

    # Apply the optimization pass to simplify the model
    onnx_model = onnx.load("model1.onnx")
    optimized_model = optimize(onnx_model)

    # Save the optimized model
    onnx.save(optimized_model, "model1.onnx")


if __name__ == '__main__':
    main()
