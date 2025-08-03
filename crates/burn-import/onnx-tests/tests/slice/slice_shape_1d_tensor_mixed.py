#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/slice/slice_shape_1d_tensor_mixed.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

OPSET_VERSION = 18

def main():
    # Create multiple sub-graphs for different test cases
    
    # Test case 1: Shape start, 1D tensor end
    print("Creating shape_start_tensor_end model...")
    create_shape_start_tensor_end_model()
    
    # Test case 2: 1D tensor start, shape end  
    print("Creating tensor_start_shape_end model...")
    create_tensor_start_shape_end_model()
    
    print("All models created successfully!")

def create_shape_start_tensor_end_model():
    """Create model with shape as starts and 1D tensor as ends"""
    
    # Create input/output value infos
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [6, 8, 10]
    )
    shape_tensor = helper.make_tensor_value_info(
        'shape_input', TensorProto.FLOAT, [2, 3]  # Will use shape [2, 3] as starts
    )
    end_tensor = helper.make_tensor_value_info(
        'ends', TensorProto.INT64, [2]  # 1D tensor with 2 elements for ends
    )
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [3, 5, 10]  # expected output shape
    )
    
    # Create axes constant (which dimensions to slice)
    axes_const = helper.make_tensor(
        name='axes',
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[0, 1]  # slice along dimensions 0 and 1
    )
    
    # Create nodes
    # 1. Shape node: shape_input -> shape
    shape_node = helper.make_node(
        'Shape',
        inputs=['shape_input'],
        outputs=['shape'],
        name='shape_op'
    )
    
    # 2. Slice node using shape as starts and ends tensor
    slice_node = helper.make_node(
        'Slice',
        inputs=['input', 'shape', 'ends', 'axes'],
        outputs=['output'],
        name='slice'
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[shape_node, slice_node],
        name='shape_start_tensor_end',
        inputs=[input_tensor, shape_tensor, end_tensor],
        outputs=[output_tensor],
        initializer=[axes_const]
    )
    
    # Create the model
    model = helper.make_model(
        graph,
        producer_name='slice_shape_1d_tensor_mixed_generator',
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    
    # Check and save
    onnx.checker.check_model(model)
    onnx.save(model, 'slice_shape_start_tensor_end.onnx')
    
    # Create test data and run inference
    session = ReferenceEvaluator(model)
    input_data = np.arange(1, 481).reshape(6, 8, 10).astype(np.float32)
    shape_input_data = np.random.randn(2, 3).astype(np.float32)
    ends = np.array([5, 8], dtype=np.int64)
    
    outputs = session.run(None, {
        'input': input_data,
        'shape_input': shape_input_data,
        'ends': ends
    })
    
    print("  Model: slice_shape_start_tensor_end.onnx")
    print(f"  Inputs: input{list(input_data.shape)}, shape_input{list(shape_input_data.shape)}, ends{list(ends.shape)}")
    print(f"  Shape extracted: {shape_input_data.shape} = [2, 3]")
    print(f"  Slice: starts=shape[2, 3], ends={ends.tolist()}")
    print(f"  Output shape: {outputs[0].shape}")
    print()

def create_tensor_start_shape_end_model():
    """Create model with 1D tensor as starts and shape as ends"""
    
    # Create input/output value infos
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [10, 12, 8]
    )
    start_tensor = helper.make_tensor_value_info(
        'starts', TensorProto.INT64, [2]  # 1D tensor with 2 elements for starts
    )
    shape_tensor = helper.make_tensor_value_info(
        'shape_input', TensorProto.FLOAT, [6, 10]  # Will use shape [6, 10] as ends
    )
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [4, 7, 8]  # expected output shape
    )
    
    # Create axes constant (which dimensions to slice)
    axes_const = helper.make_tensor(
        name='axes',
        data_type=TensorProto.INT64,
        dims=[2],
        vals=[0, 1]  # slice along dimensions 0 and 1
    )
    
    # Create nodes
    # 1. Shape node: shape_input -> shape
    shape_node = helper.make_node(
        'Shape',
        inputs=['shape_input'],
        outputs=['shape'],
        name='shape_op'
    )
    
    # 2. Slice node using starts tensor and shape as ends
    slice_node = helper.make_node(
        'Slice',
        inputs=['input', 'starts', 'shape', 'axes'],
        outputs=['output'],
        name='slice'
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[shape_node, slice_node],
        name='tensor_start_shape_end',
        inputs=[input_tensor, start_tensor, shape_tensor],
        outputs=[output_tensor],
        initializer=[axes_const]
    )
    
    # Create the model
    model = helper.make_model(
        graph,
        producer_name='slice_shape_1d_tensor_mixed_generator',
        opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    
    # Check and save
    onnx.checker.check_model(model)
    onnx.save(model, 'slice_tensor_start_shape_end.onnx')
    
    # Create test data and run inference
    session = ReferenceEvaluator(model)
    input_data = np.arange(1, 961).reshape(10, 12, 8).astype(np.float32)
    starts = np.array([2, 3], dtype=np.int64)
    shape_input_data = np.random.randn(6, 10).astype(np.float32)
    
    outputs = session.run(None, {
        'input': input_data,
        'starts': starts,
        'shape_input': shape_input_data
    })
    
    print("  Model: slice_tensor_start_shape_end.onnx")
    print(f"  Inputs: input{list(input_data.shape)}, starts{list(starts.shape)}, shape_input{list(shape_input_data.shape)}")
    print(f"  Shape extracted: {shape_input_data.shape} = [6, 10]")
    print(f"  Slice: starts={starts.tolist()}, ends=shape[6, 10]")
    print(f"  Output shape: {outputs[0].shape}")


if __name__ == '__main__':
    main()