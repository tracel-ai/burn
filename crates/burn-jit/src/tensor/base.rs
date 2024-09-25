use crate::element::JitElement;
use crate::kernel::{launch_unary, unary_op, UnaryOp};
use crate::JitRuntime;
use burn_tensor::Shape;
use cubecl::client::ComputeClient;
use cubecl::frontend::Numeric;
use cubecl::linalg::tensor::TensorHandle;
use cubecl::prelude::{TensorHandleRef, *};
use cubecl::server::Handle;
use std::marker::PhantomData;

/// The basic tensor primitive struct.
#[derive(new)]
pub struct JitTensor<R, E>
where
    R: JitRuntime,
    E: JitElement,
{
    /// Compute client for the [runtime](JitRuntime).
    pub client: ComputeClient<R::Server, R::Channel>,
    /// The buffer where the data are stored.
    pub handle: Handle<R::Server>,
    /// The shape of the tensor.
    pub shape: Shape,
    /// The device of the tensor.
    pub device: R::Device,
    /// The strides of the tensor.
    pub strides: Vec<usize>,
    pub(crate) elem: PhantomData<E>,
}

impl<R: JitRuntime, E: JitElement> From<JitTensor<R, E>> for TensorHandle<R, E> {
    fn from(val: JitTensor<R, E>) -> Self {
        TensorHandle::new(val.shape.dims.to_vec(), val.strides.to_vec(), val.handle)
    }
}

impl<R, E> core::fmt::Debug for JitTensor<R, E>
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

impl<R, E> Clone for JitTensor<R, E>
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
            strides: self.strides.clone(),
            elem: PhantomData,
        }
    }
}

impl<R, E> JitTensor<R, E>
where
    R: JitRuntime,
    E: JitElement,
{
    /// Create a new tensor with a contiguous memory layout.
    pub fn new_contiguous(
        client: ComputeClient<R::Server, R::Channel>,
        device: R::Device,
        shape: Shape,
        handle: Handle<R::Server>,
    ) -> Self {
        let ndims = shape.num_dims();
        let mut strides = vec![0; ndims];

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
            strides: self.strides.clone(),
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

    /// Return the reference to a tensor argument.
    pub fn as_tensor_arg<'a>(&'a self, vectorisation: u8) -> TensorArg<'a, R> {
        let handle: TensorHandleRef<'a, R> = self.as_handle_ref();

        unsafe {
            TensorArg::from_raw_parts(handle.handle, handle.strides, handle.shape, vectorisation)
        }
    }

    pub(crate) fn can_mut_broadcast(&self, rhs: &Self) -> bool {
        if !self.handle.can_mut() {
            return false;
        }
        let ndims = self.shape.num_dims();

        for i in 0..ndims {
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
            execute::expand::<C>(context, tensor)
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
        is_contiguous(&self.shape.dims, &self.strides)
    }
}

pub(crate) fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.is_empty() {
        return true;
    }

    if shape.len() == 1 {
        return strides[0] == 1;
    }

    let mut prev_stride = 1;
    let mut current_num_elems_shape = 1;

    for (i, (stride, shape)) in strides.iter().zip(shape).rev().enumerate() {
        if i > 0 {
            if current_num_elems_shape != *stride {
                return false;
            }

            if prev_stride >= *stride {
                return false;
            }
        }

        current_num_elems_shape *= shape;
        prev_stride = *stride;
    }

    true
}

#[cfg(test)]
mod tests {
    use crate::tensor::base::is_contiguous;

    #[test]
    fn is_contiguous_basic() {
        assert!(is_contiguous(&[32, 32], &[32, 1]));
    }

    #[test]
    fn is_contiguous_permuted() {
        assert!(!is_contiguous(&[32, 32], &[1, 32]));
    }

    #[test]
    fn is_contiguous_slice() {
        assert!(!is_contiguous(&[32, 1, 64], &[32, 64, 1]));
    }

    #[test]
    fn is_contiguous_4d_positive() {
        assert!(is_contiguous(&[8, 256, 32, 32], &[262144, 1024, 32, 1]));
    }

    #[test]
    fn is_contiguous_4d_negative() {
        assert!(!is_contiguous(&[256, 8, 32, 32], &[1024, 262144, 32, 1]));
    }
}
