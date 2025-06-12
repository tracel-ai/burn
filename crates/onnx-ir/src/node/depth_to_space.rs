use crate::{ArgType, TensorType, ir::Node};

/// Update output type for DepthToSpace operation (rank 4).
pub fn depth_to_space_update_outputs(node: &mut Node) {
    log::debug!("DepthToSpace rank inference for node {}", &node.name);

    // Extract the input tensor type to determine rank and shape
    let tensor = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("DepthToSpace: only tensor input is valid"),
    };
    assert_eq!(
        tensor.rank, 4,
        "DepthToSpace: only rank 4 tensors are supported"
    );

    // Get the block size from attribute
    let block_size = node
        .attrs
        .get("blocksize")
        .cloned()
        .expect("DepthToSpace: blocksize attribute not found")
        .into_i64() as usize;

    log::debug!(
        "DepthToSpace blocksize from attribute for {}: {:?}",
        &node.name,
        block_size
    );

    // Infer static shape based on rank and block size
    let static_shape = tensor.static_shape.clone().map(|shape| {
        let [b, c, h, w] = shape
            .try_into()
            .expect("DepthToSpace: input tensor rank is not 4");
        vec![
            b,
            c / (block_size * block_size),
            h * block_size,
            w * block_size,
        ]
    });

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        elem_type: tensor.elem_type.clone(),
        rank: tensor.rank,
        static_shape,
    });
}

/// Mode for DepthToSpace operation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DepthToSpaceMode {
    DCR,
    CRD,
}

impl From<&str> for DepthToSpaceMode {
    fn from(val: &str) -> Self {
        match val {
            "DCR" => Self::DCR,
            "CRD" => Self::CRD,
            _ => panic!("Unexpected value for DepthToSpace mode: {val}"),
        }
    }
}

/// Configuration for DepthToSpace operation
#[derive(Debug, Clone)]
pub struct DepthToSpaceConfig {
    pub mode: DepthToSpaceMode,
    pub block_size: usize,
}

impl DepthToSpaceConfig {
    /// Create a new DepthToSpaceConfig
    pub fn new(mode: DepthToSpaceMode, block_size: usize) -> Self {
        Self { mode, block_size }
    }
}

/// Create a DepthToSpaceConfig from the attributes of the node
pub fn depth_to_space_config(node: &Node) -> DepthToSpaceConfig {
    let mut block_size: Option<usize> = None;
    let mut mode = DepthToSpaceMode::DCR;

    for (key, value) in node.attrs.iter() {
        match key.as_str() {
            "blocksize" => block_size = Some(value.clone().into_i64() as usize),
            "mode" => mode = value.clone().into_string().as_str().into(),
            _ => panic!("Unexpected attribute for DepthToSpace: {key}"),
        }
    }

    let block_size = block_size.expect("DepthToSpace: blocksize must be provided");
    assert!(
        block_size > 0,
        "DepthToSpace: block_size must be greater than 0"
    );

    DepthToSpaceConfig { mode, block_size }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ElementType;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    /// Helper function to create test nodes with different repeat values
    fn create_test_node(
        rank: usize,
        static_shape: Option<Vec<usize>>,
        block_size: i64,
        mode: Option<&str>,
    ) -> Node {
        let mut builder = NodeBuilder::new(NodeType::DepthToSpace, "test_depth_to_space")
            .input_tensor_f32("input", rank, static_shape)
            .output_tensor_f32("output", rank, None) // Same rank as input
            .attr_int("blocksize", block_size);

        // Add mode attribute if provided
        if let Some(mode_str) = mode {
            builder = builder.attr_string("mode", mode_str);
        }

        builder.build()
    }

    #[test]
    fn test_basic_config() {
        let node = create_test_node(4, None, 2, None);
        let config = depth_to_space_config(&node);

        assert_eq!(config.block_size, 2);
        assert_eq!(config.mode, DepthToSpaceMode::DCR);
    }

    #[test]
    fn test_dcr_config() {
        let node = create_test_node(4, None, 3, Some("DCR"));
        let config = depth_to_space_config(&node);

        assert_eq!(config.block_size, 3);
        assert_eq!(config.mode, DepthToSpaceMode::DCR);
    }

    #[test]
    fn test_crd_config() {
        let node = create_test_node(4, None, 3, Some("CRD"));
        let config = depth_to_space_config(&node);

        assert_eq!(config.block_size, 3);
        assert_eq!(config.mode, DepthToSpaceMode::CRD);
    }

    #[test]
    fn test_static_shape_update_outputs() {
        let mut node = create_test_node(4, Some(vec![2, 4, 2, 3]), 2, None);
        depth_to_space_update_outputs(&mut node);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.static_shape, vec![2, 1, 4, 6].into());
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
