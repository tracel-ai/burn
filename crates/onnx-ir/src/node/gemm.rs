use crate::ir::{ArgType, Node, TensorType};
use core::cmp::max;

/// Update output shape for Gemm operation based on input ranks.
pub fn gemm_output_shape(node: &mut Node) {
    log::debug!("Gemm rank inference for node {}", node.name);

    let a_rank = match &node.inputs[0].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        _ => panic!("Input A should be a tensor!"),
    };
    let b_rank = match &node.inputs[1].ty {
        ArgType::Tensor(tensor) => tensor.rank,
        _ => panic!("Input B should be a tensor!"),
    };

    log::debug!(
        "Gemm input ranks for {}: a_rank={}, b_rank={}",
        node.name,
        a_rank,
        b_rank
    );

    let output_rank = max(a_rank, b_rank);
    log::debug!("Gemm output rank for {}: {}", node.name, output_rank);

    node.outputs[0].ty = ArgType::Tensor(TensorType {
        rank: output_rank,
        static_shape: None,
        elem_type: match &node.inputs[0].ty {
            ArgType::Tensor(t) => t.elem_type.clone(),
            _ => panic!("Unexpected type for input A"),
        },
    });
}

pub fn gemm_config(curr: &Node) -> (f32, f32, i64, i64) {
    let mut alpha: f32 = 1.0;
    let mut beta: f32 = 1.0;
    let mut trans_a: i64 = 0;
    let mut trans_b: i64 = 0;

    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "alpha" => alpha = value.clone().into_f32(),
            "beta" => beta = value.clone().into_f32(),
            "transA" => trans_a = value.clone().into_i64(),
            "transB" => trans_b = value.clone().into_i64(),
            _ => panic!("Unexpected attribute for Gemm: {key}"),
        }
    }

    (alpha, beta, trans_a, trans_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(
        alpha: Option<f32>,
        beta: Option<f32>,
        trans_a: Option<i64>,
        trans_b: Option<i64>,
    ) -> Node {
        let mut builder = NodeBuilder::new(NodeType::Gemm, "test_gemm")
            .input_tensor_f32("A", 2, None)
            .input_tensor_f32("B", 2, None)
            .input_tensor_f32("C", 2, None)
            .output_tensor_f32("Y", 2, None);

        if let Some(alpha_val) = alpha {
            builder = builder.attr_float("alpha", alpha_val);
        }
        if let Some(beta_val) = beta {
            builder = builder.attr_float("beta", beta_val);
        }
        if let Some(trans_a_val) = trans_a {
            builder = builder.attr_int("transA", trans_a_val);
        }
        if let Some(trans_b_val) = trans_b {
            builder = builder.attr_int("transB", trans_b_val);
        }

        builder.build()
    }

    #[test]
    fn test_gemm_config_defaults() {
        let node = create_test_node(None, None, None, None);
        let (alpha, beta, trans_a, trans_b) = gemm_config(&node);
        assert_eq!(alpha, 1.0);
        assert_eq!(beta, 1.0);
        assert_eq!(trans_a, 0);
        assert_eq!(trans_b, 0);
    }

    #[test]
    fn test_gemm_config_with_attrs() {
        let node = create_test_node(Some(2.0), Some(3.0), Some(1), Some(1));
        let (alpha, beta, trans_a, trans_b) = gemm_config(&node);
        assert_eq!(alpha, 2.0);
        assert_eq!(beta, 3.0);
        assert_eq!(trans_a, 1);
        assert_eq!(trans_b, 1);
    }

    #[test]
    fn test_gemm_config_partial_attrs() {
        let node = create_test_node(Some(0.5), None, Some(1), None);
        let (alpha, beta, trans_a, trans_b) = gemm_config(&node);
        assert_eq!(alpha, 0.5);
        assert_eq!(beta, 1.0); // default
        assert_eq!(trans_a, 1);
        assert_eq!(trans_b, 0); // default
    }
}
