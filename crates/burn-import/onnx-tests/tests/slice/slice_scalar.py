#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/slice/slice_scalar.onnx

import onnx
from onnx import helper, TensorProto


def main():
    # Create input/output value infos
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [5, 3]
    )
    start_scalar = helper.make_tensor_value_info(
        'start', TensorProto.INT64, []  # scalar
    )
    end_scalar = helper.make_tensor_value_info(
        'end', TensorProto.INT64, []  # scalar
    )
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [3, 3]  # expected output shape
    )
    
    # Create axes and steps constants
    axes_const = helper.make_tensor(
        name='axes',
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[0]  # slice along dimension 0
    )
    
    steps_const = helper.make_tensor(
        name='steps', 
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[1]  # step size 1
    )
    
    # Create the slice node directly
    slice_node = helper.make_node(
        'Slice',
        inputs=['input', 'start', 'end', 'axes', 'steps'],
        outputs=['output'],
        name='slice'
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[slice_node],
        name='slice_scalar',
        inputs=[input_tensor, start_scalar, end_scalar],
        outputs=[output_tensor],
        initializer=[axes_const, steps_const]
    )
    
    # Create the model
    model = helper.make_model(graph, producer_name='slice_scalar_generator')
    model.opset_import[0].version = 18
    
    # Check and save
    onnx.checker.check_model(model)
    onnx.save(model, 'slice_scalar.onnx')
    
    print("Finished exporting model to slice_scalar.onnx")
    print("Model structure:")
    print(f"  Inputs: input[5,3], start[], end[]")
    print(f"  Outputs: output[3,3]")
    print(f"  Nodes: {len(model.graph.node)} (just Slice)")
    print("Test case: input[1:4, :] -> output[3, 3]")


if __name__ == '__main__':
    main()