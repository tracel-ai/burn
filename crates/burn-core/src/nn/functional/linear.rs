use crate::tensor::Tensor;
use crate::tensor::backend::Backend;

/// Applies a linear transformation to the input tensor using the given weight and bias.
///
/// ```math
/// y = x @ weight + [bias]
/// ```
///
/// # Arguments:
///
/// - `input` is the input tensor, ``[..., d_input]``.
/// - `weight` is the weight tensor, ``[d_input, d_output]``.
/// - `b` is the bias tensor (optional), ``[d_output]``.
///
/// # Returns:
///
/// The transformed tensor, ``[..., d_output]``.
pub fn linear<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    weight: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
) -> Tensor<B, D> {
    if D == 1 {
        // Insert and remove an extra batch dimension for the batch matmul to work.
        return linear::<B, 2>(input.unsqueeze(), weight, bias).flatten(0, 1);
    }

    let weight = weight.unsqueeze();
    let output = input.matmul(weight);
    match bias {
        Some(bias) => output + bias.unsqueeze(),
        None => output,
    }
}

/// PyTorch-compat linear function.
///
/// Pytorch's linear function is defined with a transposed weight matrix:
///
/// ```math
/// y = x @ weight^T + [bias]
/// ```
///
/// See: https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html
///
/// # Arguments:
///
/// - `input` is the input tensor, ``[..., d_input]``.
/// - `weight` is the weight tensor, ``[d_output, d_input]``.
/// - `b` is the bias tensor (optional), ``[d_output]``.
///
/// # Returns:
///
/// The transformed tensor, ``[..., d_output]``.
pub fn linear_pytorch<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    weight: Tensor<B, 2>,
    bias: Option<Tensor<B, 1>>,
) -> Tensor<B, D> {
    linear(input, weight.transpose(), bias)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;

    #[test]
    fn test_linear_1d() {
        let weight =
            Tensor::<TestBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &Default::default());

        let x = Tensor::<TestBackend, 1>::from_data([1.0, 2.0], &Default::default());

        linear(x.clone(), weight.clone(), None)
            .into_data()
            .assert_eq(
                &Tensor::<TestBackend, 1>::from_data([7.0, 10.0], &Default::default()).into_data(),
                true,
            );

        // Compat
        linear(x.clone(), weight.clone(), None)
            .into_data()
            .assert_eq(
                &linear_pytorch(x.clone(), weight.clone().transpose(), None).into_data(),
                true,
            );
    }

    #[test]
    fn test_linear_forward_no_bias() {
        let weight =
            Tensor::<TestBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &Default::default());

        let x = Tensor::<TestBackend, 3>::from_data(
            [[[1.0, 2.0], [3.0, 4.0]], [[-1.0, -2.0], [-3.0, -4.0]]],
            &Default::default(),
        );

        linear(x.clone(), weight.clone(), None)
            .into_data()
            .assert_eq(
                &Tensor::<TestBackend, 3>::from_data(
                    [[[7.0, 10.0], [15.0, 22.0]], [[-7.0, -10.0], [-15.0, -22.0]]],
                    &Default::default(),
                )
                .into_data(),
                true,
            );

        // Compat
        linear(x.clone(), weight.clone(), None)
            .into_data()
            .assert_eq(
                &linear_pytorch(x.clone(), weight.clone().transpose(), None).into_data(),
                true,
            );
    }

    #[test]
    fn test_linear_forward_with_bias() {
        let weight =
            Tensor::<TestBackend, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &Default::default());
        let bias = Some(Tensor::<TestBackend, 1>::from_data(
            [1.0, -1.0],
            &Default::default(),
        ));

        let x = Tensor::<TestBackend, 3>::from_data(
            [[[1.0, 2.0], [3.0, 4.0]], [[-1.0, -2.0], [-3.0, -4.0]]],
            &Default::default(),
        );

        linear(x.clone(), weight.clone(), bias.clone())
            .into_data()
            .assert_eq(
                &Tensor::<TestBackend, 3>::from_data(
                    [[[8.0, 9.0], [16.0, 21.0]], [[-6.0, -11.0], [-14.0, -23.0]]],
                    &Default::default(),
                )
                .into_data(),
                true,
            );

        // Compat
        linear(x.clone(), weight.clone(), bias.clone())
            .into_data()
            .assert_eq(
                &linear_pytorch(x.clone(), weight.clone().transpose(), bias.clone()).into_data(),
                true,
            );
    }
}
