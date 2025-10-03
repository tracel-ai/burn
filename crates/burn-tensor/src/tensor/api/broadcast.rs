use burn::{prelude::Backend, tensor::Tensor};
/// Broadcast two tensors with potentially different static ranks to a common rank.
///
/// # Syntax
/// ```ignore
/// broadcast!(
///     a: Tensor<Backend, RANK_A>,
///     b: Tensor<RANK_B>
/// )
/// ```
///
/// # Parameters
/// - `a`: Identifier for the first tensor variable (e.g., `a`).
/// - `Backend`: The backend to use
/// - `RANK_A`: The static rank of the first tensor (e.g., `2`, `3`, etc.).
///
/// - `b`: Identifier for the second tensor variable (e.g., `b`).
/// - `RANK_B`: The static rank of the second tensor.
///
/// # Example
/// ```rust
///     let device = &NdArrayDevice::default();
///     type B = NdArray<f32>;
///
///     let a = Tensor::<B, 3>::from_data(
///         [
///             [[2, 8, 7, 2], [9, 14, 13, 12], [9, 14, 13, 12]],
///             [[2, 8, 7, 2], [9, 14, 13, 12], [9, 14, 13, 12]],
///         ],
///         device,
///     );
///
///     let b = Tensor::<B, 2>::from_data([[4, 11, 10, 5]], device);
///
///     let (a, b) = broadcast!(a:Tensor<B, 3>, b:Tensor<2>);
///
///     let a_add_b = a.add(b);
///
/// // Output:
/// // Tensor {
/// //   data:
/// // [[[ 6.0, 19.0, 17.0,  7.0],
/// //   [13.0, 25.0, 23.0, 17.0],
/// //   [13.0, 25.0, 23.0, 17.0]],
/// //  [[ 6.0, 19.0, 17.0,  7.0],
/// //   [13.0, 25.0, 23.0, 17.0],
/// //   [13.0, 25.0, 23.0, 17.0]]],
/// //   shape:  [2, 3, 4],
/// //   device:  Cpu,
/// //   backend:  "ndarray",
/// //   kind:  "Float",
/// //   dtype:  "f32",
/// // }
/// ```
#[macro_export]
macro_rules! broadcast {
    (
        $a:ident : Tensor<$backend:ty, $dims1:tt>,
        $b:ident : Tensor<$dims2:tt>
    ) => {{
        use $crate::broadcast::broadcast_op;
        const fn max(a: usize, b: usize) -> usize {
            if a > b { a } else { b }
        }

        const N: usize = max($dims1, $dims2);

        broadcast_op::<$backend, N, $dims1, $dims2>(&$a, &$b)
    }};
}

pub fn broadcast_op<B: Backend, const N: usize, const DA: usize, const DB: usize>(
    a: &Tensor<B, DA>,
    b: &Tensor<B, DB>,
) -> (Tensor<B, N>, Tensor<B, N>) {
    // pad left with 1s

    let a = a.clone().unsqueeze::<N>();
    let b = b.clone().unsqueeze::<N>();

    let b_shape = b.shape().dims::<N>();

    // Convert dims, change non 1 values to -1 and 1 values to corresponding tensor shape
    // for burn expand format

    // Make changes in b dimensions to match a dimensions and insert -1s

    let b_shape_new: Vec<i64> = a
        .shape()
        .dims::<N>()
        .iter_mut()
        .enumerate()
        .map(
            |(i, val)| {
                if b_shape[i] == 1 { *val as i64 } else { -1_i64 }
            },
        )
        .collect();

    // Make changes in a dimensions to match b dimensions and insert -1s

    let a_shape = a.shape().dims::<N>();

    let a_shape_new: Vec<i64> = b
        .shape()
        .dims::<N>()
        .iter_mut()
        .enumerate()
        .map(
            |(i, val)| {
                if a_shape[i] == 1 { *val as i64 } else { -1_i64 }
            },
        )
        .collect();

    // Expand both tensors to match each other using the new shapes by
    // expanding tensors a and b using new shape with -1s inserted

    let b = b.expand::<N, [i64; N]>(b_shape_new.try_into().unwrap());
    let a = a.expand::<N, [i64; N]>(a_shape_new.try_into().unwrap());

    (a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::{NdArray, NdArrayDevice};

    #[test]
    fn test_broadcast_multi_dims() {
        let device = &NdArrayDevice::default();
        type B = NdArray<f32>;

        let a = Tensor::<B, 6>::empty([7, 6, 2, 3, 1, 9], device);
        let b = Tensor::<B, 4>::empty([2, 1, 7, 1], device);

        let (a, b) = broadcast!(a: Tensor<B, 6>, b: Tensor<4>);

        assert_eq!(a.shape(), b.shape());
    }

    #[test]
    fn test_broadcast_multi_dims_values() {
        let device = &NdArrayDevice::default();
        type B = NdArray<f32>;

        let a = Tensor::<B, 3>::from_data(
            [
                [[2, 8, 7, 2], [9, 14, 13, 12], [9, 14, 13, 12]],
                [[2, 8, 7, 2], [9, 14, 13, 12], [9, 14, 13, 12]],
            ],
            device,
        );

        let b = Tensor::<B, 2>::from_data([[4, 11, 10, 5]], device);

        let (a, b) = broadcast!(a:Tensor<B, 3>, b:Tensor<2>);
        let a_add_b = a.add(b);

        Tensor::<B, 3>::from_data(
            [
                [
                    [6.0, 19.0, 17.0, 7.0],
                    [13.0, 25.0, 23.0, 17.0],
                    [13.0, 25.0, 23.0, 17.0],
                ],
                [
                    [6.0, 19.0, 17.0, 7.0],
                    [13.0, 25.0, 23.0, 17.0],
                    [13.0, 25.0, 23.0, 17.0],
                ],
            ],
            device,
        )
        .into_data()
        .assert_eq(&a_add_b.to_data(), true);
    }

    #[test]
    fn test_max_broadcast() {
        let device = &NdArrayDevice::default();
        type B = NdArray<f32>;

        let a = Tensor::<B, 1>::from_data([3.0, 2.0, 6.0, 3.0], device);
        let b = Tensor::<B, 1>::from_data([1.0, 0.5, 4.0, 7.0], device);
        let a = a.reshape([-1, 1]);

        let (a, b) = broadcast!(a:Tensor<B, 2>, b:Tensor<1>);
        let max_a_b = a.max_pair(b);

        Tensor::<B, 2>::from_data(
            [
                [3.0, 3.0, 4.0, 7.0],
                [2.0, 2.0, 4.0, 7.0],
                [6.0, 6.0, 6.0, 7.0],
                [3.0, 3.0, 4.0, 7.0],
            ],
            device,
        )
        .into_data()
        .assert_eq(&max_a_b.to_data(), true);
    }

    #[test]
    fn test_add_broadcast() {
        let device = &NdArrayDevice::default();
        type B = NdArray<f32>;

        let a = Tensor::<B, 1>::from_data([1.1, 2.2, 3.3], device);
        let b = Tensor::<B, 1>::from_data([4.0, 5.0, 6.0, 7.0], device);
        let a = a.reshape([-1, 1]);

        let (a, b) = broadcast!(a:Tensor<B, 2>, b:Tensor<1>);
        let add_a_b = a.add(b);

        Tensor::<B, 2>::from_data(
            [
                [5.1, 6.1, 7.1, 8.1],
                [6.2, 7.2, 8.2, 9.2],
                [7.3, 8.3, 9.3, 10.3],
            ],
            device,
        )
        .into_data()
        .assert_eq(&add_a_b.to_data(), true);

        let a = Tensor::<B, 1>::from_data([1.1, 2.2, 3.3], device);
        let b = Tensor::<B, 1>::from_data([4.0, 5.0, 6.0, 7.0], device);

        let b = b.reshape([-1, 1]);
        let (a, b) = broadcast!(a:Tensor<B, 1>, b:Tensor<2>);
        let add_a_b = a.add(b);

        Tensor::<B, 2>::from_data(
            [
                [5.1, 6.2, 7.3],
                [6.1, 7.2, 8.3],
                [7.1, 8.2, 9.3],
                [8.1, 9.2, 10.3],
            ],
            device,
        )
        .into_data()
        .assert_eq(&add_a_b.to_data(), true);
    }

    #[test]
    fn test_max_broadcast_uneven() {
        let device = &NdArrayDevice::default();
        type B = NdArray<f32>;

        let a = Tensor::<B, 1>::from_data([3.0, 2.0, 6.0, 3.0], device);
        let b = Tensor::<B, 1>::from_data([1.0, 0.5, 4.0, 7.0, 8.0], device);

        let b = b.reshape([-1, 1]);
        let (a, b) = broadcast!(a:Tensor<B, 1>, b:Tensor<2>);
        let max_a_b = a.max_pair(b);

        Tensor::<B, 2>::from_data(
            [
                [3.0, 2.0, 6.0, 3.0],
                [3.0, 2.0, 6.0, 3.0],
                [4.0, 4.0, 6.0, 4.0],
                [7.0, 7.0, 7.0, 7.0],
                [8.0, 8.0, 8.0, 8.0],
            ],
            device,
        )
        .into_data()
        .assert_eq(&max_a_b.to_data(), true);
    }

    #[test]
    fn test_add_broadcast_diff_dims() {
        let device = &NdArrayDevice::default();
        type B = NdArray<f32>;

        let a = Tensor::<B, 2>::from_data(
            [
                [3.0, 2.0, 6.0, 3.0],
                [3.0, 2.0, 6.0, 3.0],
                [8.0, 7.0, 7.0, 13.0],
            ],
            device,
        );

        let b = Tensor::<B, 1>::from_data([1.0, 0.5, 4.0, 7.0], device);
        let (a, b) = broadcast!(a:Tensor<B, 2>, b:Tensor<1>);

        let add_a_b = a.add(b);

        Tensor::<B, 2>::from_data(
            [
                [4.0, 2.5, 10.0, 10.0],
                [4.0, 2.5, 10.0, 10.0],
                [9.0, 7.5, 11.0, 20.0],
            ],
            device,
        )
        .into_data()
        .assert_eq(&add_a_b.to_data(), true);
    }
}
