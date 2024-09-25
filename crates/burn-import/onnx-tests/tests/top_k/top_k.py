import numpy as np
import onnx
from onnx import helper, TensorProto

# Define the input tensor
X = np.array([[0, 1, 2, 3],
              [4, 5, 6, 7],
              [8, 9, 10, 11]], dtype=np.float32)

# Define the value of K
k = 3
K = np.array([k], dtype=np.int64)
axis = -1
new_dims = [X.shape[0], k]

input_tensors = [
    helper.make_tensor_value_info('X', TensorProto.FLOAT, X.shape),
    #helper.make_tensor_value_info('K', TensorProto.INT32, K.shape)
]

output_tensors = [
    helper.make_tensor_value_info('Values', TensorProto.FLOAT, new_dims),
    helper.make_tensor_value_info('Indices', TensorProto.INT32, new_dims)
]
    
# Create the TopK node
node = helper.make_node(
    'TopK',
    inputs=['X'],# 'K'],
    outputs=['Values', 'Indices'],
    axis=axis,  # Axis along which to find the top K elements
    #largest=-1,
    k=k
)

# Create the graph
graph = helper.make_graph(
    nodes = [node],
    name = 'TopKGraph',
    inputs = input_tensors,
    outputs = output_tensors
)

# Create the model
model = helper.make_model(
    graph,
    ir_version=8,
    opset_imports=[onnx.helper.make_operatorsetid("", 1)]
)

# Check the model
onnx.checker.check_model(model)

# Save the model to a file
onnx.save(model, 'top_k.onnx')

print("Model saved to topk_model.onnx")
