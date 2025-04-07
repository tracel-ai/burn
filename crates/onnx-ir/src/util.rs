use crate::ir::{ArgType, Node};

pub fn shape_config(curr: &Node) -> (usize, usize) {
    if curr.inputs.len() != 1 {
        panic!(
            "Shape: multiple inputs are not supported (got {:?})",
            curr.inputs.len()
        );
    }

    // Extract the shape of the input tensor
    let tensor = match curr.inputs.first().unwrap().clone().ty {
        ArgType::Tensor(tensor) => tensor,
        _ => panic!("Only tensor input is valid"),
    };

    // Default: all axes up to the last one (included)
    let mut start_dim: i64 = 0;
    let mut end_dim: i64 = tensor.rank as i64;

    // Extract the attributes
    for (key, value) in curr.attrs.iter() {
        match key.as_str() {
            "start" => start_dim = value.clone().into_i64(),
            "end" => end_dim = value.clone().into_i64(),
            _ => {}
        }
    }

    // If dim is negative, it is counted from the end
    if start_dim < 0 {
        start_dim += tensor.rank as i64;
    }
    if end_dim < 0 {
        end_dim += tensor.rank as i64;
    }

    (start_dim as usize, end_dim as usize)
}

/// Infer convolution kernel shape from weight
pub fn infer_conv_kernel_shape(w: &ArgType) -> Vec<i64> {
    if let ArgType::Tensor(tensor) = w {
        // Weight [out_channels, in_channels, kernel size...]
        let shape = &tensor.shape.as_ref().unwrap()[2..];
        shape.iter().map(|x| *x as i64).collect()
    } else {
        panic!("Cannot infer kernel shape");
    }
}

#[cfg(test)]
mod tests {

    use crate::ir::{ElementType, TensorType};

    use super::*;

    #[test]
    fn test_infer_conv_kernel_shape() {
        let tensor = TensorType {
            elem_type: ElementType::Float32,
            rank: 4,
            shape: Some(vec![16, 64, 3, 3]),
        };
        let shape = infer_conv_kernel_shape(&ArgType::Tensor(tensor));

        assert_eq!(shape, vec![3, 3])
    }
}
