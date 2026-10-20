use burn_backend::{
    Shape, StreamId, TensorMetadata, distributed::ReduceOperation, ops::EmulatedCollectives,
};
use burn_ir::TensorIr;
use burn_router::{BackendRouter, RouterChannel, RouterTensor, get_client};

use super::{
    RemoteClient,
    runner::{RemoteBridge, RemoteDevice, RemoteTensorHandle, session_order},
    service,
};
use crate::shared::{LocalAllReducePart, LocalCollectiveId};

/// A local channel with direct connection to the backend runner clients.
pub struct RemoteChannel;

impl RouterChannel for RemoteChannel {
    type Device = RemoteDevice;
    type Bridge = RemoteBridge;
    type Client = RemoteClient;

    fn name(device: &Self::Device) -> String {
        format!("remote-{device:?}")
    }

    fn init_client(device: &Self::Device) -> Self::Client {
        RemoteClient::init(device.clone())
    }

    fn get_tensor_handle(tensor: &TensorIr, client: &Self::Client) -> RemoteTensorHandle {
        RemoteTensorHandle {
            client: client.clone(),
            tensor: tensor.clone(),
        }
    }

    fn register_tensor(
        _client: &Self::Client,
        _handle: RemoteTensorHandle,
        _shape: Shape,
        _dtype: burn_backend::DType,
    ) -> RouterTensor<Self::Client> {
        // This function is normally only used to move a tensor from a device to another.
        //
        // In other words, to change the client.
        panic!("Can't register manually a tensor on a remote channel.");
    }

    fn change_client_backend(
        tensor: RouterTensor<Self::Client>,
        target_device: &Self::Device, // target device
    ) -> RouterTensor<Self::Client> {
        // Get tensor handle from current client
        let original_client = tensor.client.clone();
        let desc = tensor.into_ir();
        let handle = Self::get_tensor_handle(&desc, &original_client);

        let handle = handle.change_backend(target_device);

        let id = handle.tensor.id;

        let target_client = get_client::<Self>(target_device);
        let router_tensor: RouterTensor<RemoteClient> =
            RouterTensor::new(id, handle.tensor.shape, handle.tensor.dtype, target_client);

        router_tensor
    }

    fn float_all_reduce(
        tensors: Vec<RouterTensor<RemoteClient>>,
        op: ReduceOperation,
    ) -> Vec<RouterTensor<RemoteClient>> {
        let devices: Vec<&RemoteDevice> =
            tensors.iter().map(|tensor| &tensor.client.device).collect();
        match AllReducePlan::of(&devices) {
            AllReducePlan::OnServer => Self::all_reduce_on_server(tensors, op),
            AllReducePlan::Emulated => {
                EmulatedCollectives::<BackendRouter<Self>>::float_all_reduce(tensors, op)
            }
        }
    }
}

impl RemoteChannel {
    /// Sends each member's part to its session; the server runs the all-reduce once every part
    /// has arrived.
    fn all_reduce_on_server(
        tensors: Vec<RouterTensor<RemoteClient>>,
        op: ReduceOperation,
    ) -> Vec<RouterTensor<RemoteClient>> {
        let collective = LocalCollectiveId::random();
        let members = tensors.len() as u32;
        let stream_id = StreamId::current();
        let _order = session_order();
        tensors
            .into_iter()
            .zip(0..)
            .map(|(tensor, member)| {
                let client = tensor.client.clone();
                let (shape, dtype) = (tensor.shape(), tensor.dtype());
                let out = service::new_tensor_id();
                let part = LocalAllReducePart {
                    tensor: tensor.into_ir(),
                    out,
                    collective,
                    member,
                    members,
                    op,
                };
                client
                    .handle
                    .submit(move |s| s.all_reduce_local(stream_id, part));
                client.handle.flush_queue();
                RouterTensor::new(out, shape, dtype, client)
            })
            .collect()
    }
}

/// Where an all-reduce over remote members runs.
#[derive(Debug, PartialEq, Eq)]
enum AllReducePlan {
    /// On the members' server, as its own all-reduce, NCCL on CUDA.
    OnServer,
    /// With transfers between the members.
    Emulated,
}

impl AllReducePlan {
    /// On the server when the members are distinct devices of one, with their sessions open. A
    /// member repeating a device would wait in its own session for its next part.
    fn of(devices: &[&RemoteDevice]) -> Self {
        let Some(first) = devices.first() else {
            return Self::Emulated;
        };
        let distinct = devices.iter().enumerate().all(|(i, device)| {
            devices[..i]
                .iter()
                .all(|other| other.device_index() != device.device_index())
        });
        let on_one_server = devices
            .iter()
            .all(|device| device.peer_id() == first.peer_id() && !device.session_ended());
        match distinct && on_one_server {
            true => Self::OnServer,
            false => Self::Emulated,
        }
    }
}

impl Clone for RemoteChannel {
    fn clone(&self) -> Self {
        RemoteChannel
    }
}

#[cfg(all(test, feature = "websocket"))]
mod tests {
    use super::*;
    use crate::Credential;
    use burn_communication::Address;
    use service::RemoteEndpoint;

    fn device(server: &str, index: usize) -> RemoteDevice {
        let endpoint = RemoteEndpoint::WebSocket {
            address: Address::from(server),
            credential: Credential::default(),
        };
        RemoteDevice::register(endpoint, index)
    }

    #[test]
    fn distinct_devices_of_one_server_all_reduce_there() {
        let devices = [
            device("ws://10.0.0.1:3000", 1),
            device("ws://10.0.0.1:3000", 0),
        ];

        assert_eq!(
            AllReducePlan::of(&[&devices[0], &devices[1]]),
            AllReducePlan::OnServer
        );
    }

    #[test]
    fn a_repeated_device_or_a_second_server_is_emulated() {
        let first = device("ws://10.0.0.1:3000", 0);
        let other_server = device("ws://10.0.0.2:3000", 1);

        assert_eq!(
            AllReducePlan::of(&[&first, &first]),
            AllReducePlan::Emulated
        );
        assert_eq!(
            AllReducePlan::of(&[&first, &other_server]),
            AllReducePlan::Emulated
        );
    }
}
