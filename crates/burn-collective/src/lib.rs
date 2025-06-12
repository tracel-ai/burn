pub mod backend;
pub mod aggregator;

#[cfg(all(test))]
mod tests {
    use burn_ndarray::{NdArray, NdArrayDevice};
    use burn_tensor::{Distribution, Tensor};

    use crate::backend::register;

    pub fn run_peer(peer_count: u32) {
        let input_shape = [peer_count as usize, 28, 28];
        let device = NdArrayDevice::default();

        register::<NdArray>(peer_count);

        let tensor = Tensor::<NdArray, 3>::random(input_shape, Distribution::Default, &device);

        let x = tensor.sum();

        println!("{:?}", x);
    }

    #[test]
    pub fn dummy_test() {
        const PEER_COUNT: u32 = 8;
        for _ in 0..PEER_COUNT {
            std::thread::spawn(move || run_peer(PEER_COUNT));
        }
    }
}
