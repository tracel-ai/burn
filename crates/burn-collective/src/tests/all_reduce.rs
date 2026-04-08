mod tests {
    use std::sync::mpsc::SyncSender;

    use burn_backend::ops::FloatTensorOps;
    use burn_backend::{Backend, TensorData, Tolerance};
    use burn_std::rand::get_seeded_rng;

    use serial_test::serial;

    use crate::tests::read_tensor;
    use crate::{AllReduceStrategy, PeerId, ReduceOperation};

    #[cfg(not(all(
        feature = "test-cuda",
        feature = "test-wgpu",
        feature = "test-metal",
        feature = "test-vulkan"
    )))]
    pub type TestBackend = burn_ndarray::NdArray;

    #[cfg(feature = "test-cuda")]
    pub type TestBackend = burn_cuda::CudaDevice;

    #[cfg(any(feature = "test-wgpu", feature = "test-metal", feature = "test-vulkan"))]
    pub type TestBackend = burn_wgpu::WgpuDevice<f32>;

    use crate::{CollectiveConfig, all_reduce, register, reset_collective};

    pub fn run_peer<B: Backend>(
        id: PeerId,
        config: CollectiveConfig,
        input: TensorData,
        op: ReduceOperation,
        output: SyncSender<B::FloatTensorPrimitive>,
    ) {
        let device = B::Device::default();

        register::<B>(id, device.clone(), config).unwrap();

        let tensor = B::float_from_data(input, &device);

        let tensor = all_reduce::<B>(id, tensor, op).unwrap();

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
                    burn_backend::Distribution::Default,
                    &mut get_seeded_rng(),
                )
            })
            .collect();

        let device = <TestBackend as Backend>::Device::default();

        let mut expected_tensor =
            TestBackend::float_zeros(shape.into(), &device, burn_backend::FloatDType::F32);
        for item in input.iter().take(thread_count) {
            let input_tensor = TestBackend::float_from_data(item.clone(), &device);
            expected_tensor = TestBackend::float_add(expected_tensor, input_tensor);
        }
        if op == ReduceOperation::Mean {
            expected_tensor =
                TestBackend::float_div_scalar(expected_tensor, (thread_count as u32).into());
        }

        let expected = read_tensor::<TestBackend>(expected_tensor);

        (input, expected)
    }

    fn test_all_reduce<B: Backend>(
        device_count: usize,
        op: ReduceOperation,
        strategy: AllReduceStrategy,
        tensor_size: usize,
    ) {
        reset_collective::<TestBackend>();

        let (send, recv) = std::sync::mpsc::sync_channel(32);

        let shape = vec![tensor_size];

        let (input, expected) = generate_random_input(shape, op, device_count);

        let config = CollectiveConfig::default()
            .with_num_devices(device_count)
            .with_local_all_reduce_strategy(strategy);

        for id in 0..device_count {
            let send = send.clone();
            let input = input[id].clone();

            std::thread::spawn({
                let config = config.clone();
                move || run_peer::<B>(id.into(), config, input, op, send)
            });
        }

        let first = read_tensor::<B>(recv.recv().unwrap());
        for _ in 1..device_count {
            let tensor = recv.recv().unwrap();
            read_tensor::<B>(tensor).assert_eq(&first, true);
        }

        let tol: Tolerance<f32> = Tolerance::balanced();
        expected.assert_approx_eq(&first, tol);
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_centralized_sum() {
        test_all_reduce::<TestBackend>(4, ReduceOperation::Sum, AllReduceStrategy::Centralized, 4);
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_centralized_mean() {
        test_all_reduce::<TestBackend>(4, ReduceOperation::Mean, AllReduceStrategy::Centralized, 4);
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_binary_tree_sum() {
        test_all_reduce::<TestBackend>(4, ReduceOperation::Sum, AllReduceStrategy::Tree(2), 4);
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_binary_tree_mean() {
        test_all_reduce::<TestBackend>(4, ReduceOperation::Mean, AllReduceStrategy::Tree(2), 4);
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_5_tree_sum() {
        test_all_reduce::<TestBackend>(4, ReduceOperation::Sum, AllReduceStrategy::Tree(5), 4);
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_5_tree_mean() {
        test_all_reduce::<TestBackend>(4, ReduceOperation::Mean, AllReduceStrategy::Tree(5), 4);
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_ring_sum() {
        test_all_reduce::<TestBackend>(3, ReduceOperation::Sum, AllReduceStrategy::Ring, 3);
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_ring_mean() {
        test_all_reduce::<TestBackend>(3, ReduceOperation::Mean, AllReduceStrategy::Ring, 3);
    }

    #[test]
    #[serial]
    pub fn test_all_reduce_ring_irregular_sum() {
        // this should trigger the fallback algorithm when the tensor is too small.
        test_all_reduce::<TestBackend>(4, ReduceOperation::Sum, AllReduceStrategy::Ring, 3);
    }
}
