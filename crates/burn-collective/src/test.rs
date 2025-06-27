#[cfg(all(
    test,
    any(
        feature = "ndarray",
        feature = "wgpu",
        feature = "cuda",
        feature = "metal"
    )
))]
mod tests {
    use std::sync::mpsc::SyncSender;

    use burn_common::rand::get_seeded_rng;
    use burn_tensor::{Shape, Tensor, TensorData, Tolerance, backend::Backend};

    use serial_test::serial;

    #[cfg(feature = "ndarray")]
    pub type TestBackend = burn_ndarray::NdArray<f32>;

    #[cfg(feature = "cuda")]
    pub type TestBackend = burn_cuda::Cuda<f32>;

    #[cfg(feature = "wgpu")]
    pub type TestBackend = burn_wgpu::Wgpu<f32>;

    #[cfg(feature = "metal")]
    pub type TestBackend = burn_wgpu::Wgpu<f32>;

    #[cfg(feature = "vulkan")]
    pub type TestBackend = burn_wgpu::Wgpu<f32>;

    use crate::{
        AggregateKind, AggregateParams, AggregateStrategy,
        api::{all_reduce, register, reset_collective},
    };

    pub fn run_peer<B: Backend>(
        id: u32,
        peer_count: u32,
        params: AggregateParams,
        input: TensorData,
        output: SyncSender<Tensor<B, 1>>,
    ) {
        let device = B::Device::default();

        register::<B>(id, peer_count);

        let tensor = Tensor::<B, 1>::from_data(input, &device);

        let tensor = all_reduce(tensor, params);

        output.send(tensor).unwrap();
    }

    fn generate_random_input(
        shape: Shape,
        params: AggregateParams,
        peer_count: u32,
    ) -> (Vec<TensorData>, TensorData) {
        let input: Vec<TensorData> = (0..peer_count)
            .map(|_| {
                TensorData::random::<f32, _, _>(
                    shape.clone(),
                    burn_tensor::Distribution::Default,
                    &mut get_seeded_rng(),
                )
            })
            .collect();

        let device = <TestBackend as Backend>::Device::default();

        let mut expected_tensor = Tensor::<TestBackend, 1>::zeros(shape, &device);
        for item in input.iter().take(peer_count as usize) {
            let input_tensor = Tensor::<TestBackend, 1>::from_data(item.clone(), &device);
            expected_tensor = expected_tensor.add(input_tensor);
        }
        if params.kind == AggregateKind::Mean {
            expected_tensor = expected_tensor.div_scalar(peer_count);
        }

        let expected = expected_tensor.to_data();

        (input, expected)
    }

    fn test_aggregate<B: Backend>(params: AggregateParams, peer_count: u32, tensor_size: usize) {
        reset_collective::<TestBackend>();

        let (send, recv) = std::sync::mpsc::sync_channel(1);

        let shape = Shape {
            dims: vec![tensor_size],
        };
        let (input, expected) = generate_random_input(shape, params.clone(), peer_count);

        for id in 0..peer_count {
            let send = send.clone();
            let params = params.clone();
            let input = input[id as usize].clone();
            std::thread::spawn(move || run_peer::<B>(id, peer_count, params, input, send));
        }

        let first = recv.recv().unwrap().to_data();
        for _ in 1..peer_count {
            let tensor = recv.recv().unwrap();
            tensor.to_data().assert_eq(&first, true);
        }

        let tol: Tolerance<f32> = Tolerance::balanced();
        expected.assert_approx_eq(&first, tol);
    }

    #[test]
    #[serial]
    pub fn test_aggregate_centralized_sum() {
        test_aggregate::<TestBackend>(
            AggregateParams {
                kind: AggregateKind::Sum,
                strategy: AggregateStrategy::Centralized,
            },
            4,
            4,
        );
    }

    #[test]
    #[serial]
    pub fn test_aggregate_centralized_mean() {
        test_aggregate::<TestBackend>(
            AggregateParams {
                kind: AggregateKind::Mean,
                strategy: AggregateStrategy::Centralized,
            },
            4,
            4,
        );
    }

    #[test]
    #[serial]
    pub fn test_aggregate_binary_tree_sum() {
        test_aggregate::<TestBackend>(
            AggregateParams {
                kind: AggregateKind::Sum,
                strategy: AggregateStrategy::Tree(2),
            },
            4,
            4,
        );
    }

    #[test]
    #[serial]
    pub fn test_aggregate_binary_tree_mean() {
        test_aggregate::<TestBackend>(
            AggregateParams {
                kind: AggregateKind::Mean,
                strategy: AggregateStrategy::Tree(2),
            },
            4,
            4,
        );
    }

    #[test]
    #[serial]
    pub fn test_aggregate_5_tree_sum() {
        test_aggregate::<TestBackend>(
            AggregateParams {
                kind: AggregateKind::Sum,
                strategy: AggregateStrategy::Tree(5),
            },
            4,
            4,
        );
    }

    #[test]
    #[serial]
    pub fn test_aggregate_5_tree_mean() {
        test_aggregate::<TestBackend>(
            AggregateParams {
                kind: AggregateKind::Mean,
                strategy: AggregateStrategy::Tree(5),
            },
            4,
            4,
        );
    }

    #[test]
    #[serial]
    pub fn test_aggregate_ring_sum() {
        test_aggregate::<TestBackend>(
            AggregateParams {
                kind: AggregateKind::Sum,
                strategy: AggregateStrategy::Ring,
            },
            3,
            3,
        );
    }

    #[test]
    #[serial]
    pub fn test_aggregate_ring_mean() {
        test_aggregate::<TestBackend>(
            AggregateParams {
                kind: AggregateKind::Mean,
                strategy: AggregateStrategy::Ring,
            },
            3,
            3,
        );
    }

    #[test]
    #[serial]
    pub fn test_aggregate_ring_irregular_sum() {
        // this should trigger the fallback algorithm when the tensor is too small.
        test_aggregate::<TestBackend>(
            AggregateParams {
                kind: AggregateKind::Sum,
                strategy: AggregateStrategy::Ring,
            },
            4,
            3,
        );
    }
}
