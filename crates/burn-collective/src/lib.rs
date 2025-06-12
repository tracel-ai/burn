pub mod aggregator;
pub mod backend;

#[cfg(test)]
mod tests {
    use std::sync::mpsc::SyncSender;

    use burn_ndarray::{NdArray, NdArrayDevice};
    use burn_tensor::{Distribution, Tensor};

    use crate::backend::{collective_sum, register};

    pub fn run_peer(peer_count: u32, output: SyncSender<Tensor<NdArray, 3>>) {
        let input_shape = [1, 4, 4];
        let device = NdArrayDevice::default();

        register::<NdArray>(peer_count);

        let tensor = Tensor::<NdArray, 3>::random(input_shape, Distribution::Default, &device);

        let tensor = collective_sum(tensor);

        output.send(tensor).unwrap();
    }

    #[test]
    pub fn test_aggregate() {
        const PEER_COUNT: u32 = 8;
        let (send, recv) = std::sync::mpsc::sync_channel(1);

        for _ in 0..PEER_COUNT {
            let send = send.clone();
            std::thread::spawn(move || run_peer(PEER_COUNT, send));
        }

        let first = recv.recv().unwrap().to_data();
        for _ in 1..PEER_COUNT {
            let tensor = recv.recv().unwrap();
            tensor.to_data().assert_eq(&first, true);
        }
    }
}
