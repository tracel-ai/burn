#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.onnx

class GemmModel(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, transA=0, transB=0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.transA = transA
        self.transB = transB

    def forward(self, A, B, C=None):
        if self.transA:
            A = A.t()
        if self.transB:
            B = B.t()
        product = self.alpha * torch.matmul(A, B)
        if C is not None:
            product = product + self.beta * C
        return product

# Example inputs:
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = torch.randn(2, 4)

model = GemmModel(alpha=1.0, beta=1.0, transA=0, transB=0)
model.eval()

torch.onnx.export(
    model,
    (A, B, C),
    "gemm.onnx",
    opset_version=16,
    input_names=["A", "B", "C"],
    output_names=["Y"],
    dynamic_axes={"A": {0: "batch_size"}, "Y": {0: "batch_size"}}
)

print("Exported Gemm model to gemm.onnx")
