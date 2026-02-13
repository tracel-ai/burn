mod tests {
    use std::sync::mpsc::SyncSender;

    use burn_std::rand::get_seeded_rng;
    use burn_tensor::{
        Tensor, TensorData, TensorPrimitive, Tolerance,
        backend::{Backend, PeerId},
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

    use crate::{BroadcastStrategy, CollectiveConfig, broadcast, register, reset_collective};

    pub fn run_peer<B: Backend>(
        id: PeerId,
        config: CollectiveConfig,
        input: Option<TensorData>,
        output: SyncSender<Tensor<B, 1>>,
    ) {
        let device = B::Device::default();

        register::<B>(id, device.clone(), config).unwrap();

        let tensor = input.map(|data| B::float_from_data(data, &device));
        let tensor = broadcast::<B>(id, tensor).unwrap();
        let tensor = Tensor::<B, 1>::from_primitive(TensorPrimitive::Float(tensor));

        output.send(tensor).unwrap();
    }

    fn generate_random_input(shape: Vec<usize>) -> TensorData {
        TensorData::random::<f32, _, _>(
            shape.clone(),
            burn_tensor::Distribution::Default,
            &mut get_seeded_rng(),
        )
    }

    fn test_broadcast<B: Backend>(
        device_count: usize,
        strategy: BroadcastStrategy,
        tensor_size: usize,
    ) {
        reset_collective::<TestBackend>();

        let (send, recv) = std::sync::mpsc::sync_channel(32);

        let shape = vec![tensor_size];

        let input = generate_random_input(shape);

        let config = CollectiveConfig::default()
            .with_num_devices(device_count)
            .with_local_broadcast_strategy(strategy);

        for id in 0..device_count {
            // The peer #0 is the root: it sends the tensor
            let input = if id == 0 { Some(input.clone()) } else { None };

            std::thread::spawn({
                let config = config.clone();
                let send = send.clone();
                move || run_peer::<B>(id.into(), config, input, send)
            });
        }

        // Expect all peers to receive the input tensor
        let tol: Tolerance<f32> = Tolerance::balanced();
        for _ in 0..device_count {
            let tensor = recv.recv().unwrap().to_data();
            input.assert_approx_eq(&tensor, tol);
        }
    }

    #[test]
    #[serial]
    pub fn test_broadcast_centralized_sum() {
        test_broadcast::<TestBackend>(4, BroadcastStrategy::Centralized, 4);
    }

    #[test]
    #[serial]
    pub fn test_broadcast_centralized_mean() {
        test_broadcast::<TestBackend>(4, BroadcastStrategy::Centralized, 4);
    }

    #[test]
    #[serial]
    pub fn test_broadcast_binary_tree_sum() {
        test_broadcast::<TestBackend>(4, BroadcastStrategy::Tree(2), 4);
    }

    #[test]
    #[serial]
    pub fn test_broadcast_binary_tree_mean() {
        test_broadcast::<TestBackend>(4, BroadcastStrategy::Tree(2), 4);
    }

    #[test]
    #[serial]
    pub fn test_broadcast_5_tree_sum() {
        test_broadcast::<TestBackend>(4, BroadcastStrategy::Tree(5), 4);
    }

    #[test]
    #[serial]
    pub fn test_broadcast_5_tree_mean() {
        test_broadcast::<TestBackend>(4, BroadcastStrategy::Tree(5), 4);
    }
}
