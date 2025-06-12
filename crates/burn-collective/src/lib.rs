pub mod backend;
pub mod cluster;

pub mod ndarray;

#[cfg(all(test))]
mod tests {
    use burn_ndarray::{NdArray, NdArrayDevice};
    use burn_tensor::{Distribution, Tensor};

    use crate::{backend::CollectiveBackend, cluster::ClusterMetadata};

    pub fn run_peer(id: u32, peer_count: usize) {
        let input_shape = [peer_count, 28, 28];
        let device = NdArrayDevice::default();
        let cluster_metadata = ClusterMetadata { cluster_size: peer_count, master_device: 0 };
        let _ = NdArray::register(device, id, cluster_metadata);

        let tensor = Tensor::<NdArray, 3>::random(
            input_shape,
            Distribution::Default,
            &device,
        );
        
        let x = tensor.sum();

        println!("{:?}", x);
    }

    #[test]
    pub fn dummy_test() {
        const PEER_COUNT: usize = 8;
        for id in 0..PEER_COUNT {
            std::thread::spawn(move || { run_peer(id as u32, PEER_COUNT) } );
        }
    }
}

