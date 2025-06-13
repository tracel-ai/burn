pub mod aggregator;
pub mod backend;

#[cfg(test)]
mod tests {
    use std::sync::mpsc::SyncSender;

    use burn_common::rand::get_seeded_rng;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use burn_tensor::{Shape, Tensor, TensorData, Tolerance, backend::Backend};

    use burn_wgpu::Wgpu;
    use serial_test::serial;

    use crate::{
        aggregator::{AggregateKind, AggregateParams, AggregateStrategy},
        backend::{collective_mean, collective_sum, register, reset_collective},
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

        let tensor = match params.kind {
            AggregateKind::Sum => collective_sum(tensor, params.strategy),
            AggregateKind::Mean => collective_mean(tensor, params.strategy),
        };

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

        let mut expected_tensor = Tensor::<NdArray, 1>::empty(shape, &NdArrayDevice::default());
        for i in 0..peer_count as usize {
            let input_tensor =
                Tensor::<NdArray, 1>::from_data(input[i].clone(), &NdArrayDevice::default());
            expected_tensor = expected_tensor.add(input_tensor);
        }
        if params.kind == AggregateKind::Mean {
            expected_tensor = expected_tensor.div_scalar(peer_count);
        }

        let expected = expected_tensor.to_data();

        (input, expected)
    }

    fn test_aggregate<B: Backend>(params: AggregateParams, peer_count: u32) {
        reset_collective::<NdArray>();

        let (send, recv) = std::sync::mpsc::sync_channel(1);

        let shape = Shape { dims: vec![3] };
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
    pub fn test_aggregate_sum() {
        test_aggregate::<NdArray>(
            AggregateParams {
                kind: AggregateKind::Sum,
                strategy: AggregateStrategy::Centralized,
            },
            4,
        );
    }

    #[test]
    #[serial]
    pub fn test_aggregate_mean() {
        test_aggregate::<NdArray>(
            AggregateParams {
                kind: AggregateKind::Mean,
                strategy: AggregateStrategy::Centralized,
            },
            4,
        );
    }

    #[test]
    #[serial]
    pub fn test_aggregate_binary_tree_sum() {
        test_aggregate::<NdArray>(
            AggregateParams {
                kind: AggregateKind::Sum,
                strategy: AggregateStrategy::Tree(2),
            },
            4,
        );
    }

    #[test]
    #[serial]
    pub fn test_aggregate_binary_tree_mean() {
        test_aggregate::<NdArray>(
            AggregateParams {
                kind: AggregateKind::Mean,
                strategy: AggregateStrategy::Tree(2),
            },
            4,
        );
    }

    #[test]
    #[serial]
    pub fn test_aggregate_5_tree_sum() {
        test_aggregate::<NdArray>(
            AggregateParams {
                kind: AggregateKind::Sum,
                strategy: AggregateStrategy::Tree(5),
            },
            4,
        );
    }

    #[test]
    #[serial]
    pub fn test_aggregate_5_tree_mean() {
        test_aggregate::<NdArray>(
            AggregateParams {
                kind: AggregateKind::Mean,
                strategy: AggregateStrategy::Tree(5),
            },
            4,
        );
    }

    #[test]
    #[serial]
    pub fn test_aggregate_ring_sum() {
        test_aggregate::<NdArray>(
            AggregateParams {
                kind: AggregateKind::Sum,
                strategy: AggregateStrategy::Ring,
            },
            3,
        );
    }

    #[test]
    #[serial]
    pub fn test_aggregate_ring_irregular_sum() {
        test_aggregate::<NdArray>(
            AggregateParams {
                kind: AggregateKind::Sum,
                strategy: AggregateStrategy::Ring,
            },
            4,
        );
    }

    #[test]
    #[serial]
    pub fn test_aggregate_ring_wgpu() {
        test_aggregate::<Wgpu>(
            AggregateParams {
                kind: AggregateKind::Mean,
                strategy: AggregateStrategy::Ring,
            },
            3,
        );
    }
}
