use super::{
    from_onnx::OnnxGraphIO,
    ir::{ArgType, Argument, AttributeValue, Data, Node, NodeType, TensorType},
};

/// Remap node type using kernel shape
pub fn remap_node_with_kernel_shape<F>(node: &mut Node, new_node_type: F)
where
    F: FnOnce(&Vec<i64>) -> NodeType,
{
    if let AttributeValue::Int64s(ints) = node.attrs.get("kernel_shape").unwrap() {
        node.node_type = new_node_type(ints);
    } else {
        panic!("kernel_shape is not an int64s");
    }
}

/// Remap node type to a more specific one
pub fn remap_node_type(node: &mut Node) {
    match node.node_type {
        NodeType::Conv => remap_node_with_kernel_shape(node, |ints| match ints.len() {
            1 => NodeType::Conv1d,
            2 => NodeType::Conv2d,
            _ => panic!("Only conv 1d and 2d are supported"),
        }),
        NodeType::ConvTranspose => remap_node_with_kernel_shape(node, |ints| match ints.len() {
            1 => NodeType::ConvTranspose1d,
            2 => NodeType::ConvTranspose2d,
            _ => panic!("Only conv_transpose 1d and 2d are supported"),
        }),
        NodeType::MaxPool => remap_node_with_kernel_shape(node, |ints| match ints.len() {
            1 => NodeType::MaxPool1d,
            2 => NodeType::MaxPool2d,
            _ => panic!("Only max_pool 1d and 2d are supported"),
        }),
        NodeType::AveragePool => remap_node_with_kernel_shape(node, |ints| match ints.len() {
            1 => NodeType::AveragePool1d,
            2 => NodeType::AveragePool2d,
            _ => panic!("Only avg_pool 1d and 2d are supported"),
        }),
        _ => (),
    }
}

/// Remap the unsqueeze node to a reshape node
pub(crate) fn remap_unsqueeze_to_reshape(node: &mut Node, graph_io: &mut OnnxGraphIO) {
    match graph_io.get_type(&node.outputs[0]) {
        ArgType::Tensor(output_tensor) => {
            let inner = output_tensor
                .shape
                .clone()
                .unwrap()
                .into_iter()
                .map(|x| x as i64)
                .collect::<Vec<i64>>();
            let shape_len = inner.len();
            let new_rhs_value = Some(Data::Int64s(inner));
            //moving the remap to here
            let rhs_arg = Argument {
                name: format!("{}_generated_const", node.name),
                ty: ArgType::Tensor(TensorType {
                    elem_type: super::ir::ElementType::Int64,
                    dim: 1,
                    shape: Some(vec![shape_len]),
                }),
                value: new_rhs_value,
                passed: false,
            };
            // ? should this replace the old input (reuse the old key) or should it be a new key
            // going with new key for now
            let rhs_name = rhs_arg.name.clone();
            graph_io.add_generated_const(&rhs_name, rhs_arg);
            node.inputs[1] = rhs_name;
            node.node_type = NodeType::Reshape;
        }
        _ => {}
    }
}
