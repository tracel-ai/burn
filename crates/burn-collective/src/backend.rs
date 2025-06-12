use burn_ndarray::NdArrayTensor;
use burn_tensor::{backend::Backend, Tensor};
use std::sync::mpsc::{Receiver, Sender, self};


use crate::cluster::{ClusterMetadata, ClusterOps};

pub trait CollectiveBackend: Backend {
    type Cluster: ClusterOps<Self>;

    fn get_cluster() -> Option<Self::Cluster>;
    fn get_or_init_cluster(cluster_info: ClusterMetadata) -> Self::Cluster;

    fn register(device: Self::Device, rank: u32, cluster_info: ClusterMetadata) -> Result<(), String> {
        let mut cluster = Self::get_or_init_cluster(cluster_info.clone());
        cluster.register(device, rank, cluster_info)?;

        Ok(())
    }

    fn get_device(rank: u32) -> Option<Self::Device>;
    fn get_master_device() -> Option<Self::Device>;

    fn get_cur_rank() -> Option<u32>;

    fn collective_sum<const D: usize>(tensor: Tensor<Self, D>) -> Tensor<Self, D> {
        let mut cluster = Self::get_cluster().unwrap();
        cluster.sync_op();

        let rank = Self::get_cur_rank().unwrap();
        if rank == 0 {
            // sum our tensor first
            let mut result_tensor = tensor.sum();

            // set up channel
            let (send, recv) = mpsc::channel::<NdArrayTensor<f32>>();
            cluster.set_tensor_sender(send);

            // wait for others to finish their sums
            cluster.sync_op();

            // collect all summed tensors from other ranks
            while let Ok(next_tensor) = recv.recv() {
                let next = Tensor::from_primitive(next_tensor);
                result_tensor = result_tensor.add(next);
            };

            result_tensor

        } else {
            // sum tensor and send to rank 0 device
            let tensor = tensor.sum();
            let device = Self::get_master_device().unwrap();
            tensor.to_device(&device);

            cluster.sync_op();

            let sender = cluster.get_tensor_sender().unwrap();
            sender.send(tensor);

            todo!()
        }

        // wait for 8 call a collective_sum // Semaphore
        // let tensor_distributed = DistributedBackend::from_inner(vec![tensors], device);
        // let result = tensor_distributed.sum();
        // let result = nodes.resolve(result);
    }
}
