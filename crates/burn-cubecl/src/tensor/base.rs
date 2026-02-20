use crate::CubeRuntime;
use crate::element::CubeElement;
use crate::kernel::{NumericUnaryOp, NumericUnaryOpFamily, launch_unary_numeric};
use burn_backend::quantization::QuantScheme;
use burn_backend::{DType, QTensorPrimitive, Shape, TensorMetadata};
use burn_std::{Metadata, strides, tensor::is_contiguous};
use cubecl::client::ComputeClient;
use cubecl::frontend::Numeric;
use cubecl::prelude::{TensorHandleRef, *};
use cubecl::server::Handle;
use cubecl::std::tensor::TensorHandle;
use std::marker::PhantomData;

use super::QParams;

/// The basic tensor primitive struct.
pub struct CubeTensor<R: CubeRuntime> {
    /// Compute client for the [runtime](CubeRuntime).
    pub client: ComputeClient<R>,
    /// The buffer where the data are stored.
    pub handle: Handle,
    /// The metadata of the tensor.
    pub meta: Box<Metadata>,
    /// The device of the tensor.
    pub device: R::Device,
    /// The datatype of the tensor.
    pub dtype: DType,
    /// Runtime quantization parameters, if applicable
    pub qparams: Option<QParams>,
}

impl<R: CubeRuntime> From<CubeTensor<R>> for TensorHandle<R> {
    fn from(val: CubeTensor<R>) -> Self {
        TensorHandle::new(
            val.handle,
            val.meta.shape().clone(),
            val.meta.strides().clone(),
            val.dtype.into(),
        )
    }
}

impl<R: CubeRuntime> cubecl::tune::AutotuneOutput for CubeTensor<R> {
    #[cfg(feature = "autotune-checks")]
    fn check_equivalence(&self, other: Self) {
        use crate::ops::into_data_sync;
        use burn_backend::Tolerance;

        let expected = into_data_sync::<R>(self.clone());
        let actual = into_data_sync::<R>(other);
        expected.assert_approx_eq::<f32>(&actual, Tolerance::permissive());
    }
}

impl<R> core::fmt::Debug for CubeTensor<R>
where
    R: CubeRuntime,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "CubeTensor {{ shape: {:?}, device: {:?}, strides: {:?}, elem: {}, runtime: {}}}",
            self.meta.shape(),
            self.device,
            self.meta.strides(),
            self.dtype.name(),
            R::name(&self.client),
        ))
    }
}

impl<R> Clone for CubeTensor<R>
where
    R: CubeRuntime,
{
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            handle: self.handle.clone(),
            meta: self.meta.clone(),
            device: self.device.clone(),
            dtype: self.dtype,
            qparams: self.qparams.clone(),
        }
    }
}

impl<R: CubeRuntime> TensorMetadata for CubeTensor<R> {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> Shape {
        self.meta.shape().clone()
    }

    fn rank(&self) -> usize {
        self.meta.rank()
    }
}

impl<R: CubeRuntime> QTensorPrimitive for CubeTensor<R> {
    fn scheme(&self) -> &QuantScheme {
        if let DType::QFloat(scheme) = &self.dtype {
            scheme
        } else {
            panic!(
                "Quantization scheme is not valid for dtype {:?}",
                self.dtype,
            )
        }
    }
}

impl<R> CubeTensor<R>
where
    R: CubeRuntime,
{
    /// Create a new standard tensor
    pub fn new(
        client: ComputeClient<R>,
        handle: Handle,
        metadata: Metadata,
        device: R::Device,
        dtype: DType,
    ) -> Self {
        CubeTensor {
            client,
            handle,
            meta: Box::new(metadata),
            device,
            dtype,
            qparams: None,
        }
    }

    /// Create a new tensor with a contiguous memory layout.
    pub fn new_contiguous(
        client: ComputeClient<R>,
        device: R::Device,
        shape: Shape,
        handle: Handle,
        dtype: DType,
    ) -> Self {
        let ndims = shape.num_dims();
        let mut strides = strides![0; ndims];
        let mut current = 1;

        shape.iter().enumerate().rev().for_each(|(index, val)| {
            strides[index] = current;
            current *= val;
        });

        Self {
            client,
            handle,
            meta: Box::new(Metadata::new(shape, strides)),
            device,
            dtype,
            qparams: None,
        }
    }

    /// Change the context of the current tensor and return the newly transferred tensor.
    pub fn to_client(&self, client: ComputeClient<R>, device: R::Device) -> Self {
        let desc =
            self.handle
                .copy_descriptor(self.meta.shape(), self.meta.strides(), self.elem_size());
        let alloc = self.client.to_client_tensor(desc, &client);

        Self {
            client,
            handle: alloc.handle,
            meta: Box::new(Metadata::new(self.shape(), alloc.strides)),
            device,
            dtype: self.dtype,
            qparams: self.qparams.clone(),
        }
    }

    /// Return the reference to a tensor handle.
    pub fn as_handle_ref(&self) -> TensorHandleRef<'_, R> {
        TensorHandleRef {
            handle: &self.handle,
            strides: self.meta.strides(),
            shape: self.meta.shape(),
            runtime: PhantomData,
            elem_size: self.elem_size(),
        }
    }

    /// Returns the element size of this tensor
    pub fn elem_size(&self) -> usize {
        self.dtype.size()
    }

    /// Return the reference to a tensor argument.
    pub fn as_tensor_arg<'a>(&'a self, line_size: LineSize) -> TensorArg<'a, R> {
        let size = self.dtype.size();
        let handle: TensorHandleRef<'a, R> = self.as_handle_ref();

        unsafe {
            TensorArg::from_raw_parts_and_size(
                handle.handle,
                handle.strides,
                handle.shape,
                line_size,
                size,
            )
        }
    }

    /// Return the reference to an array argument.
    pub fn as_array_arg<E: CubeElement>(&self, line_size: LineSize) -> ArrayArg<'_, R> {
        unsafe {
            ArrayArg::from_raw_parts::<E>(
                &self.handle,
                self.handle.size() as usize / core::mem::size_of::<E>(),
                line_size,
            )
        }
    }

    /// Returns the address type required to index this tensor
    pub fn required_address_type(&self) -> AddressType {
        match self.try_scheme() {
            Some(scheme) => {
                let len = self.handle.size() as usize * 8 / scheme.size_bits_value();
                AddressType::from_len(len)
            }
            None => AddressType::from_len(self.handle.size() as usize / self.dtype.size()),
        }
    }

    /// Return the `QuantScheme` if present
    pub fn try_scheme(&self) -> Option<&QuantScheme> {
        match &self.dtype {
            DType::QFloat(scheme) => Some(scheme),
            _ => None,
        }
    }

    pub(crate) fn can_mut_broadcast(&self, rhs: &Self) -> bool {
        if !self.handle.can_mut() || !self.is_nonoverlapping() {
            return false;
        }
        let ndims = self.meta.num_dims();

        for i in 0..ndims {
            let shape_lhs = self.meta.shape()[i];
            let shape_rhs = rhs.meta.shape()[i];

            // Output tensor will be different from the mutable tensor.
            if shape_lhs < shape_rhs {
                return false;
            }
        }

        true
    }

    /// Copy the current tensor.
    pub fn copy(&self) -> Self {
        struct Copy;

        #[cube]
        impl<N: Numeric> NumericUnaryOp<N> for Copy {
            type Options = ();

            fn execute(input: Line<N>, _options: &Self::Options) -> Line<N> {
                input
            }
        }

        impl NumericUnaryOpFamily for Copy {
            type Options = ();
            type Unary<N: Numeric> = Self;
        }

        let tensor = self.clone();
        launch_unary_numeric::<R, Copy, _>(tensor, |_| ())
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
    ///
    /// A tensor is contiguous if the elements are stored in memory
    /// if the strides in non-increasing order and the
    /// strides at position k is equal to the product of the shapes
    /// at all positions greater than k. However, all axes with a shape of 1 are ignored.
    pub fn is_contiguous(&self) -> bool {
        is_contiguous(self.meta.shape(), self.meta.strides())
    }

    /// Check if the current tensor has a contiguous backing buffer (no overlap and no empty memory
    /// regions within the shape).
    pub fn is_contiguous_buffer(&self) -> bool {
        self.meta.shape().num_elements() * self.dtype.size() == self.handle.size() as usize
    }

    /// Checks if the tensor is non-overlapping (can be safely written to).
    pub fn is_nonoverlapping(&self) -> bool {
        let shape = self.meta.shape();
        let strides = self.meta.strides();

        if strides.contains(&0) {
            return false;
        }
        let rank = self.rank();
        if rank > 1 {
            let mut dims = shape.iter().zip(strides.iter()).collect::<Vec<_>>();
            dims.sort_by_key(|(_, stride)| **stride);

            let mut max_offset = 0;
            for (shape, stride) in dims.into_iter() {
                if *stride <= max_offset && *shape != 1 {
                    return false;
                }

                max_offset += (*shape - 1) * *stride;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_contiguous_non_increasing() {
        assert!(is_contiguous(&[3, 1], &[1, 1]));
    }

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

    /// Based on a bug encountered in interpolate_1d
    #[test]
    fn is_contiguous_4d_unit_shape() {
        assert!(!is_contiguous(&[1, 1, 1, 9], &[72, 1, 72, 8]));
    }
}
