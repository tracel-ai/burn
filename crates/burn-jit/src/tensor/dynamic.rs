use crate::gpu::Elem;
use crate::tensor::JitTensor;
use crate::{JitElement, Runtime};
use burn_common::reader::Reader;
use burn_compute::client::ComputeClient;
use burn_compute::server::{ComputeServer, Handle};
use burn_tensor::DynRankData;
use core::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use burn_compute::channel::ComputeChannel;

pub struct DynJitTensor<S: ComputeServer, C: ComputeChannel<S>, D> {
    /// Compute client for the [runtime](Runtime).
    pub client: ComputeClient<S, C>,
    /// The buffer where the data are stored.
    pub handle: Handle<S>,
    /// The shape of the tensor.
    pub shape: Vec<usize>,
    /// The device of the tensor.
    pub device: D,
    /// The strides of the tensor.
    pub strides: Vec<usize>,
    pub elem: Elem,
}

impl<S: ComputeServer, C: ComputeChannel<S>, D: Clone> Clone for DynJitTensor<S, C, D> {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            handle: self.handle.clone(),
            shape: self.shape.clone(),
            device: self.device.clone(),
            strides: self.strides.clone(),
            elem: self.elem,
        }
    }
}

impl<S: ComputeServer, C: ComputeChannel<S>, D: Debug> Debug for DynJitTensor<S, C, D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DynJitTensor")
            .field("shape", &self.shape)
            .field("device", &self.device)
            .field("strides", &self.strides)
            .field("elem", &self.elem)
            .finish()
    }
}

impl<S: ComputeServer, C: ComputeChannel<S>, D> DynJitTensor<S, C, D> {
    pub fn new(
        client: ComputeClient<S, C>,
        device: D,
        elem: Elem,
        shape: Vec<usize>,
        handle: Handle<S>,
    ) -> Self {
        let mut strides = vec![0; shape.len()];

        let mut current = 1;
        shape.iter().enumerate().rev().for_each(|(index, val)| {
            strides[index] = current;
            current *= val;
        });

        Self {
            client,
            handle,
            shape,
            strides,
            device,
            elem,
        }
    }

    pub fn into_dyn_rank_data<E: JitElement>(self) -> Reader<DynRankData<E>> {
        assert_eq!(E::gpu_elem(), self.elem);

        self.client
            .read(&self.handle)
            .map(|bytes| DynRankData::new(E::from_bytes(&bytes).to_vec(), self.shape))
    }
}

impl<S: ComputeServer, C: ComputeChannel<S>, D: Clone> DynJitTensor<S, C, D> {
    pub fn from_dyn_rank_data<R: Runtime<Server = S, Device = D, Channel = C>, E: JitElement>(
        dyn_rank_data: DynRankData<E>,
        device: &D,
    ) -> Self {
        let client = R::client(device);
        let buffer = client.create(E::as_bytes(&dyn_rank_data.value));

        Self::new(
            client,
            device.clone(),
            E::gpu_elem(),
            dyn_rank_data.shape,
            buffer,
        )
    }
}

impl<S, C: ComputeChannel<S>, Dev, E, R, const D: usize> From<DynJitTensor<S, C, Dev>> for JitTensor<R, E, D>
where
    S: ComputeServer,
    E: JitElement,
    R: Runtime<Server = S, Device = Dev, Channel = C>,
{
    fn from(value: DynJitTensor<S, C, Dev>) -> Self {
        JitTensor {
            client: value.client,
            handle: value.handle,
            shape: value.shape.try_into().unwrap(),
            device: value.device,
            strides: value.strides.try_into().unwrap(),
            elem: PhantomData,
        }
    }
}

impl<S, C: ComputeChannel<S>, Dev, E, R, const D: usize> From<JitTensor<R, E, D>> for DynJitTensor<S, C, Dev>
where
    S: ComputeServer,
    E: JitElement,
    R: Runtime<Server = S, Device = Dev, Channel = C>,
{
    fn from(value: JitTensor<R, E, D>) -> Self {
        DynJitTensor {
            client: value.client,
            handle: value.handle,
            shape: value.shape.dims.into(),
            device: value.device,
            strides: value.strides.into(),
            elem: E::gpu_elem(),
        }
    }
}
