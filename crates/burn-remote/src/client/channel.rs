use burn_backend::{
    Shape, StreamId, TensorMetadata, distributed::ReduceOperation, ops::EmulatedCollectives,
};
use burn_ir::TensorIr;
use burn_router::{BackendRouter, RouterChannel, RouterTensor, get_client};
use std::collections::BTreeSet;

use super::{
    RemoteClient,
    runner::{RemoteBridge, RemoteDevice, RemoteTensorHandle, session_order},
    service,
};
use crate::shared::{LocalAllReducePart, LocalCollectiveId, SessionId};

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
        match Self::runs_on_server(&devices) {
            true => Self::all_reduce_on_server(tensors, op),
            false => EmulatedCollectives::<BackendRouter<Self>>::float_all_reduce(tensors, op),
        }
    }
}

impl RemoteChannel {
    /// Whether the members are distinct devices of one server with their sessions open, so the
    /// server runs the all-reduce. A member repeating a device would wait in its own session for
    /// its next part.
    fn runs_on_server(devices: &[&RemoteDevice]) -> bool {
        let Some(first) = devices.first() else {
            return false;
        };
        let indices: BTreeSet<usize> = devices.iter().map(|device| device.device_index()).collect();
        indices.len() == devices.len()
            && devices
                .iter()
                .all(|device| device.peer_id() == first.peer_id() && !device.session_ended())
    }

    /// Sends each member's part to its session; the server runs the all-reduce once every part
    /// has arrived.
    fn all_reduce_on_server(
        tensors: Vec<RouterTensor<RemoteClient>>,
        op: ReduceOperation,
    ) -> Vec<RouterTensor<RemoteClient>> {
        let collective = LocalCollectiveId::random();
        let sessions: Vec<SessionId> = tensors
            .iter()
            .map(|tensor| service::session_of(tensor.client.device.id))
            .collect();
        let stream_id = StreamId::current();
        let _order = session_order(tensors.iter().map(|tensor| &tensor.client.device));
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
                    sessions: sessions.clone(),
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

        assert!(RemoteChannel::runs_on_server(&[&devices[0], &devices[1]]));
    }

    #[test]
    fn a_repeated_device_or_a_second_server_is_emulated() {
        let first = device("ws://10.0.0.1:3000", 0);
        let other_server = device("ws://10.0.0.2:3000", 1);

        assert!(!RemoteChannel::runs_on_server(&[&first, &first]));
        assert!(!RemoteChannel::runs_on_server(&[&first, &other_server]));
    }
}
