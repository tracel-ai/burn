use std::sync::mpsc::Sender;

use burn_ndarray::NdArrayTensor;
use burn_tensor::backend::Backend;


// TODO is "Cluster" a good name? Maybe "CollectiveGroup" is better

/// Operations on a cluster, which is the struct used by the collective backend extention to keep 
/// track of the other units during collective operations.
pub trait ClusterOps<B: Backend> {
    fn register(&mut self, device: B::Device, rank: u32, cluster_info: ClusterMetadata) -> Result<(), String>;
    fn sync_op(&self);

    // TODO make generic for any backend
    fn set_tensor_sender(&mut self, sender: Sender<NdArrayTensor<f32>>);
    fn get_tensor_sender(&self) -> Option<Sender<NdArrayTensor<f32>>>;
}


#[derive(Debug, Clone)]
pub struct ClusterMetadata {
    pub cluster_size: usize,
    pub master_device: u32,
}

