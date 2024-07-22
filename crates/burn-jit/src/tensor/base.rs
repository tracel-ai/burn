use crate::element::JitElement;
use crate::kernel::{launch_unary, unary_op, UnaryOp};
use crate::JitRuntime;
use burn_tensor::Shape;
use cubecl::client::ComputeClient;
use cubecl::frontend::Numeric;
use cubecl::linalg::tensor::{matrix_layout, MatrixLayout, TensorHandle};
use cubecl::prelude::{TensorHandleRef, *};
use cubecl::server::Handle;
use std::marker::PhantomData;

/// The basic tensor primitive struct.
#[derive(new)]
pub struct JitTensor<R, E, const D: usize>
where
    R: JitRuntime,
    E: JitElement,
{
    /// Compute client for the [runtime](JitRuntime).
    pub client: ComputeClient<R::Server, R::Channel>,
    /// The buffer where the data are stored.
    pub handle: Handle<R::Server>,
    /// The shape of the tensor.
    pub shape: Shape<D>,
    /// The device of the tensor.
    pub device: R::Device,
    /// The strides of the tensor.
    pub strides: [usize; D],
    pub(crate) elem: PhantomData<E>,
}

impl<R: JitRuntime, E: JitElement, const D: usize> From<JitTensor<R, E, D>>
    for TensorHandle<R, E::Primitive>
{
    fn from(val: JitTensor<R, E, D>) -> Self {
        TensorHandle::new(val.shape.dims.to_vec(), val.strides.to_vec(), val.handle)
    }
}

impl<R, E, const D: usize> core::fmt::Debug for JitTensor<R, E, D>
where
    R: JitRuntime,
    E: JitElement,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "JitTensor {{ shape: {:?}, device: {:?}, strides: {:?}, elem: {}, runtime: {}}}",
            self.shape,
            self.device,
            self.strides,
            E::type_name(),
            R::name(),
        ))
    }
}

impl<R, E, const D: usize> Clone for JitTensor<R, E, D>
where
    R: JitRuntime,
    E: JitElement,
{
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            handle: self.handle.clone(),
            shape: self.shape.clone(),
            device: self.device.clone(),
            strides: self.strides,
            elem: PhantomData,
        }
    }
}

impl<R, E, const D: usize> JitTensor<R, E, D>
where
    R: JitRuntime,
    E: JitElement,
{
    /// Create a new tensor with a contiguous memory layout.
    pub fn new_contiguous(
        client: ComputeClient<R::Server, R::Channel>,
        device: R::Device,
        shape: Shape<D>,
        handle: Handle<R::Server>,
    ) -> Self {
        let mut strides = [0; D];

        let mut current = 1;
        shape
            .dims
            .iter()
            .enumerate()
            .rev()
            .for_each(|(index, val)| {
                strides[index] = current;
                current *= val;
            });

        Self {
            client,
            handle,
            shape,
            strides,
            device,
            elem: PhantomData,
        }
    }

    /// Change the context of the current tensor and return the newly transferred tensor.
    pub fn to_client(
        &self,
        client: ComputeClient<R::Server, R::Channel>,
        device: R::Device,
    ) -> Self {
        let bytes = burn_common::reader::try_read_sync(
            self.client.read_async(self.handle.clone().binding()),
        )
        .expect("Can only change client synchronously");
        let handle = client.create(&bytes);

        Self {
            client,
            handle,
            shape: self.shape.clone(),
            strides: self.strides,
            device,
            elem: PhantomData,
        }
    }

    /// Return the reference to a tensor handle.
    pub fn as_handle_ref(&self) -> TensorHandleRef<'_, R> {
        TensorHandleRef {
            handle: &self.handle,
            strides: &self.strides,
            shape: &self.shape.dims,
        }
    }

    pub(crate) fn can_mut_broadcast(&self, rhs: &Self) -> bool {
        if !self.handle.can_mut() {
            return false;
        }

        for i in 0..D {
            let shape_lhs = self.shape.dims[i];
            let shape_rhs = rhs.shape.dims[i];

            // Output tensor will be different from the mutable tensor.
            if shape_lhs < shape_rhs {
                return false;
            }
        }

        true
    }

    /// Copy the current tensor.
    pub fn copy(&self) -> Self {
        unary_op!(numeric(self.clone()) => |context, tensor| {
            #[cube]
            fn execute<C: Numeric>(input: C) -> C {
                input
            }
            execute::__expand::<C>(context, tensor)
        })
    }

    /// Check if the tensor is safe to mutate.
    pub fn can_mut(&self) -> bool {
        self.handle.can_mut()
    }

    /// Assert that both tensors are on the same device.
    pub fn assert_is_on_same_device(&self, other: &Self) {
        if self.device != other.device {
            panic!(
                "Both tensors should be on the same device {:?} != {:?}",
                self.device, other.device
            );
        }
    }

    /// Check if the current tensor is contiguous.
    pub fn is_contiguous(&self) -> bool {
        self.matrix_layout() == MatrixLayout::Contiguous
    }

    pub(crate) fn matrix_layout(&self) -> MatrixLayout {
        matrix_layout(&self.strides)
    }
}
