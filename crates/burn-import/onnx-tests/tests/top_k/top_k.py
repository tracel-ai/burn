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
axis = 1
new_dims = [X.shape[0], k]

def create_model(op_set_version: int):
    input_tensors = [helper.make_tensor_value_info('X', TensorProto.FLOAT, X.shape)]

    output_tensors = [
        helper.make_tensor_value_info('Values', TensorProto.FLOAT, new_dims),
        helper.make_tensor_value_info('Indices', TensorProto.INT32, new_dims)
    ]

    # Create the TopK node
    if op_set_version > 1:
        node = helper.make_node(
            'TopK',
            inputs=['X', 'K'],
            outputs=['Values', 'Indices'],
            axis=axis,  # Axis along which to find the top K elements
        )
        input_tensors.append(helper.make_tensor_value_info('K', TensorProto.INT32, K.shape))
    else:
        node = helper.make_node(
            'TopK',
            inputs=['X'],
            outputs=['Values', 'Indices'],
            axis=axis,  # Axis along which to find the top K elements
            k=k
        )

    # Create the graph
    graph = helper.make_graph(
        nodes = [node],
        name = 'TopKGraph',
        inputs = input_tensors,
        outputs = output_tensors,
        # Uncomment when initializers are supported. Currently we can't test opset 10/11 since the code will require a k value to be initialized for testing.
        #initializer = [
        #    helper.make_tensor('X', TensorProto.FLOAT, X.shape, X),
        #    helper.make_tensor('K', TensorProto.INT64, [1], [k]),
        #]
    )

    # Create the model
    model = helper.make_model(
        graph,
        ir_version=8,
        opset_imports=[onnx.helper.make_operatorsetid("", op_set_version)]
    )
    # Check the model
    onnx.checker.check_model(model)

    # Save the model to a file
    onnx.save(model, f'top_k_opset_{op_set_version}.onnx')
    print(f"Model saved to top_k_opset_{op_set_version}.onnx")
    
def main():
    # Uncomment when initializers are supported.
    # for op_set_version in [1, 10, 11]:
    for op_set_version in [1]:
        create_model(op_set_version)


if __name__ == "__main__":
    main()