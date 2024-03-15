use crate::tensor::JitTensor;
use crate::{JitElement, Runtime};
use burn_common::reader::Reader;
use burn_compute::channel::ComputeChannel;
use burn_compute::client::ComputeClient;
use burn_compute::server::{ComputeServer, Handle};
use burn_tensor::DynRankData;
use core::fmt::{Debug, Formatter};
use std::marker::PhantomData;

#[derive(Clone, Copy, Debug)]
/// The kind of element stored in a [DynJitTensor]. This refers to the elements interpretation, not the actual data stored.
/// For example, a [u32] could be used as storage for a boolean, while being an integer type.
pub enum ElemKind {
    /// A float element.
    Float,
    /// A integer element.
    Int,
    /// A boolean element.
    Bool,
}

/// The dynamic JIT tensor primitive.
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
    /// The kind of elements the tensor stores.
    pub elem_kind: ElemKind,
}

impl<S: ComputeServer, C: ComputeChannel<S>, D: Clone> Clone for DynJitTensor<S, C, D> {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            handle: self.handle.clone(),
            shape: self.shape.clone(),
            device: self.device.clone(),
            strides: self.strides.clone(),
            elem_kind: self.elem_kind,
        }
    }
}

impl<S: ComputeServer, C: ComputeChannel<S>, D: Debug> Debug for DynJitTensor<S, C, D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DynJitTensor")
            .field("shape", &self.shape)
            .field("device", &self.device)
            .field("strides", &self.strides)
            .field("elem", &self.elem_kind)
            .finish()
    }
}

impl<S: ComputeServer, C: ComputeChannel<S>, D> DynJitTensor<S, C, D> {
    /// Creates a [DynJitTensor], which should use a contiguous memory layout.
    pub fn new(
        client: ComputeClient<S, C>,
        device: D,
        elem_kind: ElemKind,
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
            elem_kind,
        }
    }

    /// Converts this dynamic tensor primitive into [DynRankData] of a given element type, which should match the element type stored in the tensor.
    pub fn into_dyn_rank_data<E: JitElement>(self) -> Reader<DynRankData<E>> {
        self.client
            .read(&self.handle)
            .map(|bytes| DynRankData::new(E::from_bytes(&bytes).to_vec(), self.shape))
    }

    /// Converts a [JitTensor] into a [DynJitTensor].
    pub fn from_jit_tensor<E, R, const DIM: usize>(
        value: JitTensor<R, E, DIM>,
        elem_kind: ElemKind,
    ) -> DynJitTensor<S, C, D>
    where
        E: JitElement,
        R: Runtime<Server = S, Device = D, Channel = C>,
    {
        DynJitTensor {
            client: value.client,
            handle: value.handle,
            shape: value.shape.dims.into(),
            device: value.device,
            strides: value.strides.into(),
            elem_kind,
        }
    }
}

impl<S: ComputeServer, C: ComputeChannel<S>, D: Clone> DynJitTensor<S, C, D> {
    /// Converts this [DynRankData] into a dynamic JIT tensor.
    pub fn from_dyn_rank_data<R: Runtime<Server = S, Device = D, Channel = C>, E: JitElement>(
        dyn_rank_data: DynRankData<E>,
        elem_kind: ElemKind,
        device: &D,
    ) -> Self {
        let client = R::client(device);
        let buffer = client.create(E::as_bytes(&dyn_rank_data.value));

        Self::new(
            client,
            device.clone(),
            elem_kind,
            dyn_rank_data.shape,
            buffer,
        )
    }
}

impl<S, C: ComputeChannel<S>, Dev, E, R, const D: usize> From<DynJitTensor<S, C, Dev>>
    for JitTensor<R, E, D>
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
