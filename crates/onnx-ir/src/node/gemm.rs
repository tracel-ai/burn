use crate::ir::Node;

pub fn gemm_config(curr: &Node) -> (f32, f32, i64, i64) {
    let alpha = curr
        .attrs
        .get("alpha")
        .map(|val| val.clone().into_f32())
        .unwrap_or(1.0);
    let beta = curr
        .attrs
        .get("beta")
        .map(|val| val.clone().into_f32())
        .unwrap_or(1.0);
    let trans_a = curr
        .attrs
        .get("transA")
        .map(|val| val.clone().into_i64())
        .unwrap_or(0);
    let trans_b = curr
        .attrs
        .get("transB")
        .map(|val| val.clone().into_i64())
        .unwrap_or(0);

    (alpha, beta, trans_a, trans_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, AttributeValue, ElementType, NodeType, TensorType};
    use std::collections::HashMap;

    fn create_test_node(
        alpha: Option<f32>,
        beta: Option<f32>,
        trans_a: Option<i64>,
        trans_b: Option<i64>,
    ) -> Node {
        let inputs = vec![
            Argument {
                name: "A".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 2,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            },
            Argument {
                name: "B".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 2,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            },
            Argument {
                name: "C".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 2,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            },
        ];

        let mut attrs = HashMap::new();
        if let Some(alpha_val) = alpha {
            attrs.insert("alpha".to_string(), AttributeValue::Float32(alpha_val));
        }
        if let Some(beta_val) = beta {
            attrs.insert("beta".to_string(), AttributeValue::Float32(beta_val));
        }
        if let Some(trans_a_val) = trans_a {
            attrs.insert("transA".to_string(), AttributeValue::Int64(trans_a_val));
        }
        if let Some(trans_b_val) = trans_b {
            attrs.insert("transB".to_string(), AttributeValue::Int64(trans_b_val));
        }

        Node {
            node_type: NodeType::Gemm,
            name: "test_gemm".to_string(),
            inputs,
            outputs: vec![Argument {
                name: "Y".to_string(),
                ty: ArgType::Tensor(TensorType {
                    elem_type: ElementType::Float32,
                    rank: 2,
                    static_shape: None,
                }),
                value: None,
                passed: true,
            }],
            attrs,
        }
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
