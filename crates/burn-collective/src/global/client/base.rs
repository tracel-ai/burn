use burn_tensor::backend::Backend;
use futures::stream::FuturesUnordered;
use std::{marker::PhantomData, sync::Arc};
use tokio::runtime::Runtime;
use tokio_util::sync::CancellationToken;

use futures_util::stream::StreamExt;

use crate::{
    GlobalAllReduceParams, GlobalRegisterParams,
    global::{
        client::{data_server::TensorDataClient, worker::GlobalClientWorker},
        shared::base::{
            CentralizedAllReduceStrategy::{self, Central, Peripheral},
            NodeAddress, RemoteRequest, RemoteResponse, RingAllReduceStrategy,
            TreeAllReduceStrategy,
        },
    },
};

pub(crate) struct GlobalCollectiveClient<B: Backend> {
    data_service: TensorDataClient<B>,
    data_client_address: Arc<NodeAddress>,
    worker: GlobalClientWorker,
    _runtime: Runtime,
    _phantom_data: PhantomData<B>,
}

impl<B: Backend> GlobalCollectiveClient<B> {
    pub fn new(server_address: &str, client_address: &str, data_server_port: u16) -> Self {
        let cancel_token = CancellationToken::new();

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        let data_client = TensorDataClient::new(&runtime, cancel_token.clone(), data_server_port);

        let worker = GlobalClientWorker::new(&runtime, cancel_token.clone(), server_address);

        Self {
            data_service: data_client,
            data_client_address: Arc::new(NodeAddress(client_address.to_owned())),
            worker,
            _runtime: runtime,
            _phantom_data: PhantomData,
        }
    }

    pub async fn register(&mut self, node_id: u32, params: GlobalRegisterParams) {
        let node_addr = self.data_client_address.as_ref().clone();
        let req = RemoteRequest::Register {
            node_id,
            node_addr,
            num_nodes: params.num_nodes,
        };
        let resp = self.worker.request(req).await;
        if resp != RemoteResponse::RegisterAck {
            panic!(
                "The response to a register request should be a RegisterAck, not {:?}",
                resp
            );
        }
    }

    pub async fn all_reduce(
        &mut self,
        tensor: B::FloatTensorPrimitive,
        params: GlobalAllReduceParams,
        device: &B::Device,
    ) -> B::FloatTensorPrimitive {
        let req = RemoteRequest::AllReduce { params };
        let resp = self.worker.request(req).await;
        match resp {
            RemoteResponse::CentralizedAllReduceStrategy(strategy) => {
                return self.centralized_all_reduce(tensor, device, strategy).await;
            }
            RemoteResponse::TreeAllReduceStrategy(strategy) => {
                return self.tree_all_reduce(tensor, device, strategy).await;
            }
            RemoteResponse::RingAllReduceStrategy(strategy) => {
                return self.ring_all_reduce(tensor, device, strategy).await;
            }
            RemoteResponse::Error(err) => panic!("Global collective server error: {err}"),
            resp => panic!(
                "The response to a register request should be a RegisterAck, not {:?}",
                resp
            ),
        };
    }

    async fn centralized_all_reduce(
        &mut self,
        tensor: B::FloatTensorPrimitive,
        device: &B::Device,
        strategy: CentralizedAllReduceStrategy,
    ) -> B::FloatTensorPrimitive {
        match strategy {
            Central { other_nodes } => {
                // download tensors from other nodes
                let mut futures = other_nodes
                    .iter()
                    .map(|x| {
                        let device = device.clone(); // if device is Clone, otherwise ref
                        let data_service = self.data_service.clone();
                        async move {
                            let data = data_service.download_next_tensor(x, 0).await.unwrap();
                            B::float_from_data(data, &device)
                        }
                    })
                    .collect::<FuturesUnordered<_>>();

                // Sum all downloads async
                let mut sum = tensor;
                while let Some(res) = futures.next().await {
                    // If the tensor is empty, we can skip it
                    sum = B::float_add(sum, res);
                }

                // Expose result
                self.data_service
                    .expose(sum.clone(), other_nodes.len() as u32, 1)
                    .await;

                sum
            }
            Peripheral { central_node } => {
                // Expose input
                self.data_service.expose(tensor, 1, 0).await;

                // Download result
                let data = self
                    .data_service
                    .download_next_tensor(&central_node, 1)
                    .await
                    .unwrap();

                B::float_from_data(data, device)
            }
        }
    }

    /// For each Send operation, we expose the tensor N times. For each Receive operation,
    /// we download the tensor from the specified address, and if it hasn't been sent yet,
    /// we combine it with the previous result. If it has been sent, we override it.
    async fn tree_all_reduce(
        &mut self,
        tensor: B::FloatTensorPrimitive,
        device: &B::Device,
        strategy: TreeAllReduceStrategy,
    ) -> B::FloatTensorPrimitive {
        // Download tensors from children async
        let mut downloads = strategy
            .children
            .iter()
            .map(|child| {
                let data_service = self.data_service.clone();
                async move {
                    let data = data_service.download_next_tensor(child, 0).await.unwrap();

                    B::float_from_data(data, device)
                }
            })
            .collect::<FuturesUnordered<_>>();

        // Sum download results
        let mut result = tensor;
        while let Some(res) = downloads.next().await {
            result = B::float_add(result, res);
        }

        // Expose the result to the parent
        if let Some(parent) = &strategy.parent {
            self.data_service.expose(result.clone(), 1, 1).await;

            // Download final tensor from parent
            let data = self
                .data_service
                .download_next_tensor(parent, 1)
                .await
                .unwrap();
            let parent_tensor = B::float_from_data(data, device);
            result = parent_tensor;
        }

        // Expose the final result to all children
        if !strategy.children.is_empty() {
            self.data_service
                .expose(result.clone(), strategy.children.len() as u32 + 1, 1)
                .await;
        }

        result
    }

    pub async fn ring_all_reduce(
        &mut self,
        _tensor: B::FloatTensorPrimitive,
        _device: &B::Device,
        _strategy: RingAllReduceStrategy,
    ) -> B::FloatTensorPrimitive {
        // Slice the tensor, should correspond to the local slicing.
        todo!()
    }

    pub async fn finish(&mut self) {
        self.worker.close_connection().await;
        self.data_service.close().await;
    }
}

impl<B: Backend> Drop for GlobalCollectiveClient<B> {
    fn drop(&mut self) {
        eprintln!("Dropping Global Collective Client");
    }
}
