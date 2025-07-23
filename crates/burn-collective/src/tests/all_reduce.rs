mod tests {
    use std::sync::mpsc::SyncSender;

    use burn_common::rand::get_seeded_rng;
    use burn_tensor::{Shape, Tensor, TensorData, Tolerance, backend::Backend};

    use serial_test::serial;

    #[cfg(feature = "test-ndarray")]
    pub type TestBackend = burn_ndarray::NdArray<f32>;

    #[cfg(feature = "test-cuda")]
    pub type TestBackend = burn_cuda::Cuda<f32>;

    #[cfg(feature = "test-wgpu")]
    pub type TestBackend = burn_wgpu::Wgpu<f32>;

    #[cfg(feature = "test-metal")]
    pub type TestBackend = burn_wgpu::Wgpu<f32>;

    #[cfg(feature = "test-vulkan")]
    pub type TestBackend = burn_wgpu::Wgpu<f32>;

    use crate::{
        AllReduceStrategy, CollectiveConfig, DeviceId, ReduceKind, all_reduce, register,
        reset_collective,
    };

    pub fn run_peer<B: Backend>(
        config: CollectiveConfig,
        input: TensorData,
        output: SyncSender<Tensor<B, 1>>,
    ) {
        let device = B::Device::default();

        register::<B>(&config).unwrap();

        let tensor = Tensor::<B, 1>::from_data(input, &device);

        let tensor = all_reduce(tensor, &config).unwrap();

        output.send(tensor).unwrap();
    }

    fn generate_random_input(
        shape: Shape,
        reduce_kind: ReduceKind,
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
        if reduce_kind == ReduceKind::Mean {
            expected_tensor = expected_tensor.div_scalar(thread_count);
        }

        let expected = expected_tensor.to_data();

        (input, expected)
    }

    fn test_all_reduce<B: Backend>(
        device_count: u32,
        reduce_kind: ReduceKind,
        strategy: AllReduceStrategy,
        tensor_size: usize,
    ) {
        reset_collective::<TestBackend>();

        let (send, recv) = std::sync::mpsc::sync_channel(32);

        let shape = Shape {
            dims: vec![tensor_size],
        };

        let (input, expected) = generate_random_input(shape, reduce_kind, device_count);
        let mut global_idx: usize = 0;

        for id in 0..device_count {
            let config = CollectiveConfig::default()
                .with_device_id(id.into())
                .with_num_devices(device_count)
                .with_all_reduce_kind(reduce_kind)
                .with_local_strategy(strategy);

            let send = send.clone();
            let input = input[global_idx].clone();

            std::thread::spawn(move || run_peer::<B>(config, input, send));

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
        test_all_reduce::<TestBackend>(4, ReduceKind::Sum, AllReduceStrategy::Centralized, 4);
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_centralized_mean() {
        test_all_reduce::<TestBackend>(4, ReduceKind::Mean, AllReduceStrategy::Centralized, 4);
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_binary_tree_sum() {
        test_all_reduce::<TestBackend>(4, ReduceKind::Sum, AllReduceStrategy::Tree(2), 4);
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_binary_tree_mean() {
        test_all_reduce::<TestBackend>(4, ReduceKind::Mean, AllReduceStrategy::Tree(2), 4);
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_5_tree_sum() {
        test_all_reduce::<TestBackend>(4, ReduceKind::Sum, AllReduceStrategy::Tree(5), 4);
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_5_tree_mean() {
        test_all_reduce::<TestBackend>(4, ReduceKind::Mean, AllReduceStrategy::Tree(5), 4);
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_ring_sum() {
        test_all_reduce::<TestBackend>(3, ReduceKind::Sum, AllReduceStrategy::Ring, 3);
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_ring_mean() {
        test_all_reduce::<TestBackend>(3, ReduceKind::Mean, AllReduceStrategy::Ring, 3);
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_ring_irregular_sum() {
        // this should trigger the fallback algorithm when the tensor is too small.
        test_all_reduce::<TestBackend>(4, ReduceKind::Sum, AllReduceStrategy::Centralized, 3);
    }
}
