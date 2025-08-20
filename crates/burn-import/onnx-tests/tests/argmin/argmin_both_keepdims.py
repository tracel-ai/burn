#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/argmin/argmin_both_keepdims.onnx

import torch
import torch.nn as nn
import onnx
import onnx.reference

class Model(nn.Module):
    def __init__(self, argmin_dim: int = 1):
        super(Model, self).__init__()
        self._argmin_dim = argmin_dim

    def forward(self, x):
        # Test both keepdim=True and keepdim=False on the same input
        y_keepdims_true = torch.argmin(input=x, dim=self._argmin_dim, keepdim=True)
        y_keepdims_false = torch.argmin(input=x, dim=self._argmin_dim, keepdim=False)
        return y_keepdims_true, y_keepdims_false

def main():
    # Export to onnx
    model = Model(1)  # argmin along dimension 1
    model.eval()
    device = torch.device("cpu")
    onnx_name = "argmin_both_keepdims.onnx"
    dummy_input = torch.randn((3, 4), device=device)
    torch.onnx.export(model, dummy_input, onnx_name,
                      verbose=False, opset_version=16,
                      output_names=['keepdims_true', 'keepdims_false'])
    
    print("Finished exporting model to {}".format(onnx_name))

    # Test with specific input to verify behavior
    test_input = torch.tensor([[3.0, 1.0, 2.0], [2.0, 4.0, 1.0]], dtype=torch.float32)
    print("Test input data shape: {}".format(test_input.shape))
    print("Test input data:\n{}".format(test_input))
    
    output_true, output_false = model.forward(test_input)
    print("Test output keepdims=True shape: {}".format(output_true.shape))
    print("Test output keepdims=True data: {}".format(output_true))
    print("Test output keepdims=False shape: {}".format(output_false.shape))
    print("Test output keepdims=False data: {}".format(output_false))
    
    print("Expected:")
    print("  keepdims=True: [[1], [2]] (shape [2, 1])")
    print("  keepdims=False: [1, 2] (shape [2])")
    
    # Verify with ONNX reference implementation
    onnx_model = onnx.load(onnx_name)
    ref_outputs = onnx.reference.ReferenceEvaluator(onnx_model).run(None, {"input1": test_input.numpy()})
    print("ONNX reference outputs:")
    print("  keepdims=True:", ref_outputs[0], "shape:", ref_outputs[0].shape)
    print("  keepdims=False:", ref_outputs[1], "shape:", ref_outputs[1].shape)
    print("PyTorch vs ONNX keepdims=True match:", torch.allclose(output_true, torch.from_numpy(ref_outputs[0])))
    print("PyTorch vs ONNX keepdims=False match:", torch.allclose(output_false, torch.from_numpy(ref_outputs[1])))

if __name__ == '__main__':
    main()