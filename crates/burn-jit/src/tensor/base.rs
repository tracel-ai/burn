use crate::codegen::dialect::gpu::{Elem, Operator, Scope, UnaryOperator};
use crate::element::JitElement;
use crate::{unary, Runtime};
use burn_compute::client::ComputeClient;
use burn_compute::server::Handle;
use burn_tensor::Shape;
use std::marker::PhantomData;

/// The basic tensor primitive struct.
pub struct JitTensor<R, E, const D: usize>
where
    R: Runtime,
    E: JitElement,
{
    /// Compute client for the [runtime](Runtime).
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

impl<R, E, const D: usize> core::fmt::Debug for JitTensor<R, E, D>
where
    R: Runtime,
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
    R: Runtime,
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
    R: Runtime,
    E: JitElement,
{
    /// Create a new tensor with a contiguous memory layout.
    pub fn new(
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
        let bytes = self
            .client
            .read(&self.handle)
            .read_sync()
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
        // Seems like using the copy buffer from the `wgpu` API leads to race condition when they
        // are used inplace afterward.
        //
        // To avoid them we need to execute the whole pipeline, which leads to significant
        // slowdowns.
        //
        // The solution is just to use a simple unary compute shader.
        unary!(
            operation: |scope: &mut Scope, elem: Elem| Operator::Assign(UnaryOperator {
                input: scope.read_array(0, elem),
                out: scope.create_local(elem),
            }),
            runtime: R,
            input: self.clone(),
            elem: E
        )
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
        let mut current_stride = 0;
        for d in 0..D {
            let stride = self.strides[D - 1 - d];

            if stride <= current_stride {
                return false;
            }

            current_stride = stride;
        }

        true
    }

    pub(crate) fn batch_swapped_with_row_col(&self) -> bool {
        for d in 0..D - 2 {
            let stride = self.strides[d];
            if stride < self.strides[D - 2] || stride < self.strides[D - 1] {
                return true;
            }
        }
        false
    }
}
