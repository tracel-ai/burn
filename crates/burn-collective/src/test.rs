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
        AllReduceParams, AllReduceStrategy, ReduceKind, RegisterParams,
        api::{all_reduce, register, reset_collective},
    };

    pub fn run_peer<B: Backend>(
        id: u32,
        reg_params: RegisterParams,
        params: AllReduceParams,
        input: TensorData,
        output: SyncSender<Tensor<B, 1>>,
    ) {
        let device = B::Device::default();

        register::<B>(id, reg_params);

        let tensor = Tensor::<B, 1>::from_data(input, &device);

        let tensor = all_reduce(tensor, params);

        output.send(tensor).unwrap();
    }

    fn generate_random_input(
        shape: Shape,
        params: AllReduceParams,
        thread_count: u32,
    ) -> (Vec<TensorData>, TensorData) {
        let input: Vec<TensorData> = (0..thread_count)
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
        for item in input.iter().take(thread_count as usize) {
            let input_tensor = Tensor::<TestBackend, 1>::from_data(item.clone(), &device);
            expected_tensor = expected_tensor.add(input_tensor);
        }
        if params.kind == ReduceKind::Mean {
            expected_tensor = expected_tensor.div_scalar(thread_count);
        }

        let expected = expected_tensor.to_data();

        (input, expected)
    }

    fn test_all_reduce<B: Backend>(device_count: u32, params: AllReduceParams, tensor_size: usize) {
        reset_collective::<TestBackend>();

        let (send, recv) = std::sync::mpsc::sync_channel(32);

        let shape = Shape {
            dims: vec![tensor_size],
        };

        let (input, expected) = generate_random_input(shape, params.clone(), device_count);
        let mut global_idx: usize = 0;
        let reg_params = RegisterParams {
            num_devices: device_count,
            global_params: None,
        };
        for id in 0..device_count {
            let send = send.clone();
            let reg_params = reg_params.clone();
            let params = params.clone();
            let input = input[global_idx].clone();
            std::thread::spawn(move || run_peer::<B>(id, reg_params, params, input, send));

            global_idx += 1;
        }

        let first = recv.recv().unwrap().to_data();
        for _ in 1..device_count {
            let tensor = recv.recv().unwrap();
            tensor.to_data().assert_eq(&first, true);
        }

        let tol: Tolerance<f32> = Tolerance::balanced();
        expected.assert_approx_eq(&first, tol);
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_centralized_sum() {
        test_all_reduce::<TestBackend>(
            4,
            AllReduceParams {
                kind: ReduceKind::Sum,
                local_strategy: AllReduceStrategy::Centralized,
                global_strategy: None,
            },
            4,
        );
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_centralized_mean() {
        test_all_reduce::<TestBackend>(
            4,
            AllReduceParams {
                kind: ReduceKind::Mean,
                local_strategy: AllReduceStrategy::Centralized,
                global_strategy: None,
            },
            4,
        );
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_binary_tree_sum() {
        test_all_reduce::<TestBackend>(
            4,
            AllReduceParams {
                kind: ReduceKind::Sum,
                local_strategy: AllReduceStrategy::Tree(2),
                global_strategy: None,
            },
            4,
        );
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_binary_tree_mean() {
        test_all_reduce::<TestBackend>(
            4,
            AllReduceParams {
                kind: ReduceKind::Mean,
                local_strategy: AllReduceStrategy::Tree(2),
                global_strategy: None,
            },
            4,
        );
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_5_tree_sum() {
        test_all_reduce::<TestBackend>(
            4,
            AllReduceParams {
                kind: ReduceKind::Sum,
                local_strategy: AllReduceStrategy::Tree(5),
                global_strategy: None,
            },
            4,
        );
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_5_tree_mean() {
        test_all_reduce::<TestBackend>(
            4,
            AllReduceParams {
                kind: ReduceKind::Mean,
                local_strategy: AllReduceStrategy::Tree(5),
                global_strategy: None,
            },
            4,
        );
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_ring_sum() {
        test_all_reduce::<TestBackend>(
            3,
            AllReduceParams {
                kind: ReduceKind::Sum,
                local_strategy: AllReduceStrategy::Ring,
                global_strategy: None,
            },
            3,
        );
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_ring_mean() {
        test_all_reduce::<TestBackend>(
            3,
            AllReduceParams {
                kind: ReduceKind::Mean,
                local_strategy: AllReduceStrategy::Ring,
                global_strategy: None,
            },
            3,
        );
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_ring_irregular_sum() {
        // this should trigger the fallback algorithm when the tensor is too small.
        test_all_reduce::<TestBackend>(
            4,
            AllReduceParams {
                kind: ReduceKind::Sum,
                local_strategy: AllReduceStrategy::Ring,
                global_strategy: None,
            },
            3,
        );
    }
}
