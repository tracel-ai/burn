mod tests {
    use std::sync::mpsc::SyncSender;

    use burn_std::rand::get_seeded_rng;
    use burn_tensor::{
        Tensor, TensorData, TensorPrimitive, Tolerance,
        backend::{Backend, PeerId, ReduceOperation},
    };

    use serial_test::serial;

    #[cfg(not(all(
        feature = "test-cuda",
        feature = "test-wgpu",
        feature = "test-metal",
        feature = "test-vulkan"
    )))]
    pub type TestBackend = burn_ndarray::NdArray<f32>;

    #[cfg(feature = "test-cuda")]
    pub type TestBackend = burn_cuda::Cuda<f32>;

    #[cfg(feature = "test-wgpu")]
    pub type TestBackend = burn_wgpu::Wgpu<f32>;

    #[cfg(feature = "test-metal")]
    pub type TestBackend = burn_wgpu::Wgpu<f32>;

    #[cfg(feature = "test-vulkan")]
    pub type TestBackend = burn_wgpu::Wgpu<f32>;

    use crate::{CollectiveConfig, ReduceStrategy, reduce, register, reset_collective};

    pub fn run_peer<B: Backend>(
        id: PeerId,
        config: CollectiveConfig,
        input: TensorData,
        op: ReduceOperation,
        root: PeerId,
        output: SyncSender<Option<Tensor<B, 1>>>,
    ) {
        let device = B::Device::default();

        register::<B>(id, device.clone(), config).unwrap();

        let tensor = Tensor::<B, 1>::from_data(input, &device);

        let tensor = tensor.into_primitive().tensor();
        let tensor = reduce::<B>(id, tensor, op, root).unwrap();
        let tensor = tensor.map(|t| Tensor::<B, 1>::from_primitive(TensorPrimitive::Float(t)));

        output.send(tensor).unwrap();
    }

    fn generate_random_input(
        shape: Vec<usize>,
        op: ReduceOperation,
        thread_count: usize,
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
        for item in input.iter().take(thread_count) {
            let input_tensor = Tensor::<TestBackend, 1>::from_data(item.clone(), &device);
            expected_tensor = expected_tensor.add(input_tensor);
        }
        if op == ReduceOperation::Mean {
            expected_tensor = expected_tensor.div_scalar(thread_count as u32);
        }

        let expected = expected_tensor.to_data();

        (input, expected)
    }

    fn test_reduce<B: Backend>(
        device_count: usize,
        op: ReduceOperation,
        strategy: ReduceStrategy,
        tensor_size: usize,
    ) {
        reset_collective::<TestBackend>();

        let (send, recv) = std::sync::mpsc::sync_channel(32);

        let shape = vec![tensor_size];

        let (input, expected) = generate_random_input(shape, op, device_count);

        let config = CollectiveConfig::default()
            .with_num_devices(device_count)
            .with_local_reduce_strategy(strategy);

        let root: PeerId = 0.into();
        for id in 0..device_count {
            let send = send.clone();
            let input = input[id as usize].clone();

            std::thread::spawn({
                let config = config.clone();
                move || run_peer::<B>(id.into(), config, input, op, root, send)
            });
        }

        let mut result = None;
        for _ in 0..device_count {
            let tensor = recv.recv().unwrap();
            if tensor.is_some() {
                if result.is_some() {
                    panic!("Two peers received the result of an reduce!");
                }
                result = tensor.map(|t| t.to_data());
            }
        }

        let tol: Tolerance<f32> = Tolerance::balanced();
        expected.assert_approx_eq(&result.expect("One peer has received the result"), tol);
    }

    #[test]
    #[serial]
    pub fn test_reduce_centralized_sum() {
        test_reduce::<TestBackend>(4, ReduceOperation::Sum, ReduceStrategy::Centralized, 4);
    }

    #[test]
    #[serial]
    pub fn test_reduce_centralized_mean() {
        test_reduce::<TestBackend>(4, ReduceOperation::Mean, ReduceStrategy::Centralized, 4);
    }

    #[test]
    #[serial]
    pub fn test_reduce_binary_tree_sum() {
        test_reduce::<TestBackend>(4, ReduceOperation::Sum, ReduceStrategy::Tree(2), 4);
    }

    #[test]
    #[serial]
    pub fn test_reduce_binary_tree_mean() {
        test_reduce::<TestBackend>(4, ReduceOperation::Mean, ReduceStrategy::Tree(2), 4);
    }

    #[test]
    #[serial]
    pub fn test_reduce_5_tree_sum() {
        test_reduce::<TestBackend>(4, ReduceOperation::Sum, ReduceStrategy::Tree(5), 4);
    }

    #[test]
    #[serial]
    pub fn test_reduce_5_tree_mean() {
        test_reduce::<TestBackend>(4, ReduceOperation::Mean, ReduceStrategy::Tree(5), 4);
    }
}
