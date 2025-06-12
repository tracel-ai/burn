// use crate::backend::CollectiveBackend;
// use crate::cluster::{ClusterMetadata, ClusterOps};
// use burn_ndarray::{NdArray, NdArrayDevice, NdArrayTensor};
// use std::collections::HashMap;
// use std::sync::mpsc::Sender;
// use std::sync::{Arc, Barrier, Mutex};
// use std::thread::{self, ThreadId};
//
// #[derive(Debug)]
// pub struct NdArrayClusterState {
//     devices: HashMap<u32, NdArrayDevice>,
//     threads: HashMap<ThreadId, u32>,
//     tensor_sender: Option<Sender<NdArrayTensor<f32>>>,
//     metadata: ClusterMetadata,
// }
//
// #[derive(Debug)]
// pub struct NdArrayCluster {
//     thread_barrier: Barrier,
//     state: Mutex<NdArrayClusterState>,
// }
//
// impl ClusterOps<NdArray> for Arc<NdArrayCluster> {
//     // TODO add a regsiter here, put the register code to CollectiveBackend trait
//     fn register(
//         &mut self,
//         device: NdArrayDevice,
//         rank: u32,
//         cluster_info: ClusterMetadata,
//     ) -> Result<(), String> {
//         {
//             let mut cluster_state = self.state.lock().unwrap();
//             if cluster_state.metadata.cluster_size != cluster_info.cluster_size {
//                 return Err(format!(
//                     "Cluster size doesn't match the previously assigned cluster size"
//                 ));
//             }
//             cluster_state.threads.insert(thread::current().id(), rank);
//             cluster_state.devices.insert(rank, device);
//         }
//
//         // Wait for other the threads to register
//         self.thread_barrier.wait();
//
//         Ok(())
//     }
//
//     fn sync_op(&self) {
//         self.thread_barrier.wait();
//     }
//
//     fn set_tensor_sender(&mut self, sender: Sender<NdArrayTensor<f32>>) {
//         let mut state = self.state.lock().unwrap();
//         state.tensor_sender = Some(sender);
//     }
//
//     fn get_tensor_sender(&self) -> Option<Sender<NdArrayTensor<f32>>> {
//         let state = self.state.lock().unwrap();
//         state.tensor_sender.as_ref().map(|x| x.clone())
//     }
// }
//
// // only one local cluster for NdArray
// static NDARRAY_CLUSTER: Mutex<Option<Arc<NdArrayCluster>>> = Mutex::new(None);
//
// fn get_ndarray_cluster() -> Option<Arc<NdArrayCluster>> {
//     let cluster = NDARRAY_CLUSTER.lock().unwrap();
//     match cluster.as_ref() {
//         Some(cluster_ref) => Some(cluster_ref.clone()),
//         None => None,
//     }
// }
//
// fn get_or_init_ndarray_cluster(cluster_info: ClusterMetadata) -> Arc<NdArrayCluster> {
//     let cluster = NDARRAY_CLUSTER.lock().unwrap();
//     match cluster.as_ref() {
//         Some(cluster_ref) => cluster_ref.clone(),
//         None => {
//             let state = NdArrayClusterState {
//                 devices: HashMap::new(),
//                 threads: HashMap::new(),
//                 metadata: cluster_info.clone(),
//                 tensor_sender: None,
//             };
//             let cluster = NdArrayCluster {
//                 thread_barrier: Barrier::new(cluster_info.cluster_size),
//                 state: Mutex::new(state),
//             };
//             Arc::new(cluster)
//         }
//     }
// }
//
// impl CollectiveBackend for NdArray {
//     type Cluster = Arc<NdArrayCluster>;
//
//     fn cluster() -> Option<Arc<NdArrayCluster>> {
//         get_ndarray_cluster()
//     }
//
//     fn get_or_init_cluster(cluster_info: ClusterMetadata) -> Arc<NdArrayCluster> {
//         get_or_init_ndarray_cluster(cluster_info)
//     }
//
//     fn get_device(rank: u32) -> Option<Self::Device> {
//         let cluster = NDARRAY_CLUSTER.lock().unwrap();
//         match cluster.as_ref() {
//             Some(cluster_ref) => {
//                 let cluster_state = cluster_ref.state.lock().unwrap();
//                 let device = cluster_state.devices.get(&rank)?.clone();
//                 Some(device)
//             }
//             None => None,
//         }
//     }
//
//     fn get_master_device() -> Option<Self::Device> {
//         let cluster = NDARRAY_CLUSTER.lock().unwrap();
//         match cluster.as_ref() {
//             Some(cluster_ref) => {
//                 let cluster_state = cluster_ref.state.lock().unwrap();
//                 let device = cluster_state
//                     .devices
//                     .get(&cluster_state.metadata.master_device)?
//                     .clone();
//                 Some(device)
//             }
//             None => None,
//         }
//     }
//
//     fn get_cur_rank() -> Option<u32> {
//         let cluster = NDARRAY_CLUSTER.lock().unwrap();
//         match cluster.as_ref() {
//             Some(cluster_ref) => {
//                 let cluster_state = cluster_ref.state.lock().unwrap();
//                 let tid = thread::current().id();
//                 let rank = cluster_state.threads.get(&tid)?;
//                 Some(*rank as u32)
//             }
//             None => None,
//         }
//     }
// }
