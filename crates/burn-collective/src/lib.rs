pub mod aggregator;
pub mod backend;

#[cfg(test)]
mod tests {
    use std::sync::mpsc::SyncSender;

    use burn_ndarray::NdArray;
    use burn_tensor::{Distribution, Tensor, backend::Backend};

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
        output: SyncSender<Tensor<B, 3>>,
    ) {
        let input_shape = [1, 8, 4];
        let device = B::Device::default();

        register::<B>(id, peer_count);

        let tensor = Tensor::<B, 3>::random(input_shape, Distribution::Default, &device);

        let tensor = match params.kind {
            AggregateKind::Sum => collective_sum(tensor, params.strategy),
            AggregateKind::Mean => collective_mean(tensor, params.strategy),
        };

        output.send(tensor).unwrap();
    }

    fn test_aggregate<B: Backend>(params: AggregateParams) {
        reset_collective::<NdArray>();
        println!("Testing with {:?}", params);

        const PEER_COUNT: u32 = 4;
        let (send, recv) = std::sync::mpsc::sync_channel(1);

        for id in 0..PEER_COUNT {
            let send = send.clone();
            let params = params.clone();
            std::thread::spawn(move || run_peer::<B>(id, PEER_COUNT, params, send));
        }

        let first = recv.recv().unwrap().to_data();
        for _ in 1..PEER_COUNT {
            let tensor = recv.recv().unwrap();
            tensor.to_data().assert_eq(&first, true);
        }
    }

    #[test]
    #[serial]
    pub fn test_aggregate_sum() {
        test_aggregate::<NdArray>(AggregateParams {
            kind: AggregateKind::Sum,
            strategy: AggregateStrategy::Centralized,
        });
    }

    #[test]
    #[serial]
    pub fn test_aggregate_mean() {
        test_aggregate::<NdArray>(AggregateParams {
            kind: AggregateKind::Mean,
            strategy: AggregateStrategy::Centralized,
        });
    }

    #[test]
    #[serial]
    pub fn test_aggregate_binary_tree_sum() {
        test_aggregate::<NdArray>(AggregateParams {
            kind: AggregateKind::Sum,
            strategy: AggregateStrategy::Tree(2),
        });
    }

    #[test]
    #[serial]
    pub fn test_aggregate_binary_tree_mean() {
        test_aggregate::<NdArray>(AggregateParams {
            kind: AggregateKind::Mean,
            strategy: AggregateStrategy::Tree(2),
        });
    }

    #[test]
    #[serial]
    pub fn test_aggregate_5_tree_sum() {
        test_aggregate::<NdArray>(AggregateParams {
            kind: AggregateKind::Sum,
            strategy: AggregateStrategy::Tree(5),
        });
    }

    #[test]
    #[serial]
    pub fn test_aggregate_5_tree_mean() {
        test_aggregate::<NdArray>(AggregateParams {
            kind: AggregateKind::Mean,
            strategy: AggregateStrategy::Tree(5),
        });
    }

    #[test]
    #[serial]
    pub fn test_aggregate_5_ring_sum() {
        test_aggregate::<NdArray>(AggregateParams {
            kind: AggregateKind::Sum,
            strategy: AggregateStrategy::Ring,
        });
    }

    #[test]
    #[serial]
    pub fn test_aggregate_5_ring_mean_wgpu() {
        test_aggregate::<Wgpu>(AggregateParams {
            kind: AggregateKind::Mean,
            strategy: AggregateStrategy::Ring,
        });
    }
}
