import torch
import torch.nn as nn

class Sampling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return a + b * 0.5

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 2)
        self.fc3 = nn.Linear(16, 2)
        self.fc4 = nn.Linear(2, 32)
        self.sampling = Sampling()

    def forward(self, x):
        x = self.fc1(x)
        middle1 = self.fc2(x)
        middle2 = self.fc3(x)
        x = self.sampling(middle1, middle2)
        x = self.fc4(x)
        return middle1, middle2, x
    
    
def main():
    # Export to onnx
    model = Model()
    model.eval()
    device = torch.device("cpu")

    file_name = "graph_multiple_output_tracking.onnx"
    input1 = torch.ones(2, 32, device=device)

    torch.onnx.export(model, (input1), file_name,
                      verbose=False, opset_version=16)

    print("Finished exporting model to {}".format(file_name))

if __name__ == '__main__':
    main()
