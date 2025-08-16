import onnx
import numpy as np
from onnx import helper, TensorProto

# Define the inputs
input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 3])

# Create an initializer for the constant we'll add
constant_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
constant_initializer = helper.make_tensor(
    name='const_tensor',
    data_type=TensorProto.FLOAT,
    dims=constant_data.shape,
    vals=constant_data.flatten()
)

# Define the Add node
add_node = helper.make_node(
    'Add',
    inputs=['input', 'const_tensor'],  # Second input is an initializer
    outputs=['output'],
    name='add_with_initializer'
)

# Define the output
output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 3])

# Create the graph
graph = helper.make_graph(
    nodes=[add_node],
    name='InitializerTest',
    inputs=[input_tensor],
    outputs=[output_tensor],
    initializer=[constant_initializer]  # This is key - makes const_tensor an initializer
)

# Create the model with proper opset version
model = helper.make_model(
    graph, 
    producer_name='test',
    opset_imports=[helper.make_operatorsetid("", 16)]
)

# Save the model
onnx.save(model, "initializer_to_const.onnx")

print("Generated initializer_to_const.onnx with initializer")