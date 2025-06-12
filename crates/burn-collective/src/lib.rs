pub mod aggregator;
pub mod backend;

#[cfg(test)]
mod tests {
    use std::sync::mpsc::SyncSender;

    use burn_ndarray::{NdArray, NdArrayDevice};
    use burn_tensor::{Distribution, Tensor};

    use serial_test::serial;

    use crate::backend::{collective_mean, collective_sum, register, reset_collective};

    pub fn run_peer_sum(id: u32, peer_count: u32, output: SyncSender<Tensor<NdArray, 3>>) {
        let input_shape = [1, 4, 4];
        let device = NdArrayDevice::default();

        register::<NdArray>(id, peer_count);

        let tensor = Tensor::<NdArray, 3>::random(input_shape, Distribution::Default, &device);

        let tensor = collective_sum(tensor);

        output.send(tensor).unwrap();
    }

    pub fn run_peer_mean(id: u32, peer_count: u32, output: SyncSender<Tensor<NdArray, 3>>) {
        let input_shape = [1, 4, 4];
        let device = NdArrayDevice::default();

        register::<NdArray>(id, peer_count);

        let tensor = Tensor::<NdArray, 3>::random(input_shape, Distribution::Default, &device);

        let tensor = collective_mean(tensor);

        output.send(tensor).unwrap();
    }

    #[test]
    #[serial]
    pub fn test_aggregate_sum() {
        reset_collective::<NdArray>();

        const PEER_COUNT: u32 = 8;
        let (send, recv) = std::sync::mpsc::sync_channel(1);

        for id in 0..PEER_COUNT {
            let send = send.clone();
            std::thread::spawn(move || run_peer_sum(id, PEER_COUNT, send));
        }

        let first = recv.recv().unwrap().to_data();
        for _ in 1..PEER_COUNT {
            let tensor = recv.recv().unwrap();
            tensor.to_data().assert_eq(&first, true);
        }
    }

    #[test]
    #[serial]
    pub fn test_aggregate_mean() {
        reset_collective::<NdArray>();

        const PEER_COUNT: u32 = 8;
        let (send, recv) = std::sync::mpsc::sync_channel(1);

        for id in 0..PEER_COUNT {
            let send = send.clone();
            std::thread::spawn(move || run_peer_mean(id, PEER_COUNT, send));
        }

        let first = recv.recv().unwrap().to_data();
        for _ in 1..PEER_COUNT {
            let tensor = recv.recv().unwrap();
            tensor.to_data().assert_eq(&first, true);
        }
    }
}
