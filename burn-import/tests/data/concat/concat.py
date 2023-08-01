# used to generate model: burn-import/tests/data/conv2d/conv2d.onnx

import torch
import torch.nn as nn
import onnx
from onnxoptimizer import optimize

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        # Concatenate along the channel dimension
        y = torch.cat((x,x), 1)
        x = torch.cat((x,y), 1)
        z = torch.cat((y,y), 1)
        x = torch.cat((x,y,z), 1)
        return x

def main():

    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")
    onnx_name = "concat.onnx"
    dummy_input = torch.randn(1,256,13,13, device=device)
    torch.onnx.export(model, dummy_input, onnx_name,
                      verbose=False, opset_version=16)

if __name__ == '__main__':
    main()
