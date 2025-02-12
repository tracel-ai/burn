use crate::{LibTorchDevice, TchElement};
use burn_tensor::{
    quantization::{
        AffineQuantization, QTensorPrimitive, QuantizationScheme, QuantizationStrategy,
        QuantizationType, SymmetricQuantization,
    },
    DType, Shape, TensorData, TensorMetadata,
};
use libc::c_void;
use std::sync::Arc;

/// A reference to a tensor storage.
///
/// We manually implement `Sync` and `Send` unsafely, so even if we could use `Rc`, it isn't safe.
#[allow(clippy::arc_with_non_send_sync)]
pub type StorageRef = Arc<*mut c_void>;

/// A reference to a tensor storage.
#[derive(PartialEq, Debug, Clone)]
pub enum Storage {
    /// When a tensor is a partial view of another tensor.
    View {
        /// Storage reference for the whole buffer.
        buffer_ref: StorageRef,
        /// Storage reference for the partial buffer.
        view_ref: StorageRef,
    },
    /// When a tensor use all of its buffer.
    Owned {
        /// Storage reference for the whole buffer.
        buffer_ref: StorageRef,
    },
}

impl Storage {
    /// Check if the storage can be used inplace.
    pub fn can_mut(&self) -> bool {
        match self {
            Storage::View {
                buffer_ref: start_ref,
                view_ref,
            } => Arc::strong_count(start_ref) == 1 && Arc::strong_count(view_ref) == 1,
            Storage::Owned {
                buffer_ref: start_ref,
            } => Arc::strong_count(start_ref) == 1,
        }
    }

    /// Get the whole buffer reference.
    pub fn buffer_ref(&self) -> &StorageRef {
        match self {
            Storage::View {
                buffer_ref: start_ref,
                view_ref: _,
            } => start_ref,
            Storage::Owned {
                buffer_ref: start_ref,
            } => start_ref,
        }
    }
}

/// A tensor using the tch backend.
#[derive(Debug, PartialEq)]
pub struct TchTensor {
    /// Handle to the tensor. Call methods on this field.
    pub tensor: tch::Tensor,

    /// The tensor's storage
    pub storage: Storage,
}

impl TensorMetadata for TchTensor {
    fn dtype(&self) -> DType {
        match self.tensor.kind() {
            tch::Kind::Uint8 => DType::U8,
            tch::Kind::Int8 => DType::I8,
            tch::Kind::Int16 => DType::I16,
            tch::Kind::Int => DType::I32,
            tch::Kind::Int64 => DType::I64,
            tch::Kind::Half => DType::F16,
            tch::Kind::Float => DType::F32,
            tch::Kind::Double => DType::F64,
            tch::Kind::Bool => DType::Bool,
            tch::Kind::QUInt8 => DType::U8,
            tch::Kind::BFloat16 => DType::BF16,
            // Complex and quantization types are not valid/implemented.
            _ => unimplemented!(),
        }
    }

    fn shape(&self) -> Shape {
        Shape::from(self.tensor.size())
    }
}

impl TchTensor {
    /// Create a new tensor.
    ///
    /// Note that if the tensor was created from an operation that may reuse the same tensor
    /// storage as the parent, you should use [from_existing](TchTensor::from_existing)
    /// instead.
    pub fn new(tensor: tch::Tensor) -> Self {
        #[allow(clippy::arc_with_non_send_sync)]
        let storage = Storage::Owned {
            buffer_ref: Arc::new(tensor.data_ptr()),
        };

        Self { tensor, storage }
    }

    /// Create a tensor that was created from an operation executed on a parent tensor.
    ///
    /// If the child tensor shared the same storage as its parent, it will be cloned, effectively
    /// tracking how much tensors point to the same memory space.
    pub fn from_existing(tensor: tch::Tensor, storage_parent: Storage) -> Self {
        let storage_child = tensor.data_ptr();
        let mut is_a_new_tensor = true;

        match &storage_parent {
            Storage::View {
                buffer_ref: start_ref,
                view_ref,
            } => {
                if storage_child == *start_ref.as_ref() || storage_child == *view_ref.as_ref() {
                    is_a_new_tensor = false;
                }
            }
            Storage::Owned {
                buffer_ref: start_ref,
            } => {
                if storage_child == *start_ref.as_ref() {
                    is_a_new_tensor = false;
                }
            }
        };

        let storage = match is_a_new_tensor {
            true => Storage::Owned {
                #[allow(clippy::arc_with_non_send_sync)]
                buffer_ref: Arc::new(storage_child),
            },
            false => storage_parent.clone(),
        };

        Self { tensor, storage }
    }

    /// Create a tensor that uses a part of its parent tensor such as slice and narrow.
    pub fn partial(tensor: tch::Tensor, storage_parent: Storage) -> Self {
        let storage = Storage::View {
            buffer_ref: storage_parent.buffer_ref().clone(),
            #[allow(clippy::arc_with_non_send_sync)]
            view_ref: Arc::new(tensor.data_ptr()),
        };
        Self { tensor, storage }
    }
}

// This is safe since we don't use autodiff from LibTorch.
// Also, atomic reference counting is used to know if the tensor's data can be reused.
// If there are multiple reference on the same tensor, it becomes read only.
unsafe impl Send for TchTensor {}
unsafe impl Sync for TchTensor {}

impl TchTensor {
    /// Checks if the tensor can be mutated in-place.
    ///
    /// Returns `true` if the tensor's stride does not contain zero (no broadcasting)
    /// and the storage can be mutated.
    pub fn can_mut(&self) -> bool {
        let stride_contains_zero = self.tensor.stride().iter().any(|&s| s == 0);

        !stride_contains_zero && self.storage.can_mut()
    }

    /// Executes an operation on a tensor if the data can be reused.
    pub fn mut_ops<F: Fn(&mut tch::Tensor) -> tch::Tensor>(
        &mut self,
        func: F,
    ) -> Option<TchTensor> {
        if !self.can_mut() {
            return None;
        }

        let data = self.storage.clone();
        Some(TchTensor::from_existing(func(&mut self.tensor), data))
    }

    /// Executes a unary operation, reusing the tensor data if possible.
    pub fn unary_ops<FOwn, FRef>(self, fown: FOwn, fref: FRef) -> TchTensor
    where
        FOwn: Fn(tch::Tensor) -> tch::Tensor,
        FRef: Fn(&tch::Tensor) -> tch::Tensor,
    {
        if !self.can_mut() {
            return TchTensor::from_existing(fref(&self.tensor), self.storage);
        }

        TchTensor::from_existing(fown(self.tensor), self.storage)
    }

    /// Executes a binary operation, reusing the tensor data if possible.
    pub fn binary_ops_tensor<FLMut, FRMut, FRef>(
        mut lhs: Self,
        mut rhs: Self,
        flmut: FLMut,
        frmut: FRMut,
        fref: FRef,
    ) -> TchTensor
    where
        FLMut: Fn(&mut tch::Tensor, &tch::Tensor) -> tch::Tensor,
        FRMut: Fn(&tch::Tensor, &mut tch::Tensor) -> tch::Tensor,
        FRef: Fn(&tch::Tensor, &tch::Tensor) -> tch::Tensor,
    {
        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();

        // Both lhs and rhs are expected to have the same rank
        let d_out = lhs_shape.num_dims();
        let mut out_shape = Shape::from(vec![1usize; d_out]);

        for i in 0..d_out {
            out_shape.dims[i] = usize::max(lhs_shape.dims[i], rhs_shape.dims[i]);
        }

        let num_elements_out = out_shape.num_elements();

        // Attempt to mutate lhs tensor
        if lhs_shape.num_elements() == num_elements_out {
            if let Some(output) = lhs.mut_ops(|lhs| flmut(lhs, &rhs.tensor)) {
                return output;
            }
        }

        // Attempt to mutate rhs tensor
        if rhs_shape.num_elements() == num_elements_out {
            if let Some(output) = rhs.mut_ops(|rhs| frmut(&lhs.tensor, rhs)) {
                return output;
            }
        }

        let storage = lhs.storage;
        let tensor = fref(&lhs.tensor, &rhs.tensor);

        TchTensor::from_existing(tensor, storage)
    }
}

impl Clone for TchTensor {
    fn clone(&self) -> Self {
        Self {
            tensor: self.tensor.shallow_clone(),
            storage: self.storage.clone(),
        }
    }
}

/// A shape that can be used by LibTorch.
#[derive(Debug)]
pub struct TchShape {
    /// The shape's dimensions.
    pub dims: Vec<i64>,
}

impl From<Shape> for TchShape {
    fn from(shape: Shape) -> Self {
        TchShape {
            dims: shape.dims.into_iter().map(|d| d as i64).collect(),
        }
    }
}

impl From<&[usize]> for TchShape {
    fn from(shape: &[usize]) -> Self {
        TchShape {
            dims: shape.iter().map(|d| *d as i64).collect(),
        }
    }
}

impl TchTensor {
    /// Creates a new tensor from a shape and a device.
    ///
    /// # Arguments
    ///
    /// * `data` - The tensor's data.
    /// * `device` - The device on which the tensor will be allocated.
    ///
    /// # Returns
    ///
    /// A new tensor.
    pub fn from_data<E: TchElement>(data: TensorData, device: tch::Device) -> Self {
        let shape_tch = TchShape::from(data.shape.as_slice());
        let tensor = tch::Tensor::from_slice(data.as_slice::<E>().unwrap()).to(device);
        let tensor = tensor.reshape(shape_tch.dims).to_kind(E::KIND);

        Self::new(tensor)
    }
}

impl TchTensor {
    /// Creates an empty tensor from a shape and a device.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// A new empty tensor.
    pub fn empty<E: tch::kind::Element>(shape: Shape, device: LibTorchDevice) -> Self {
        let shape_tch = TchShape::from(shape);
        let tensor = tch::Tensor::empty(shape_tch.dims, (E::KIND, device.into()));

        Self::new(tensor)
    }
}

/// A quantized tensor for the tch backend.
#[derive(Clone, Debug)]
pub struct TchQTensor {
    /// The quantized tensor.
    pub qtensor: TchTensor,
    /// The quantization scheme.
    pub scheme: QuantizationScheme,
}

impl TchQTensor {
    /// Returns the quantization strategy, including quantization parameters, for the given tensor.
    pub fn strategy(&self) -> QuantizationStrategy {
        match &self.scheme {
            QuantizationScheme::PerTensorAffine(dtype) => match dtype {
                QuantizationType::QInt8 => {
                    let scale = self.qtensor.tensor.q_scale();
                    let offset = self.qtensor.tensor.q_zero_point();
                    QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(
                        scale as f32,
                        offset as i8,
                    ))
                }
            },
            QuantizationScheme::PerTensorSymmetric(dtype) => match dtype {
                QuantizationType::QInt8 => {
                    let scale = self.qtensor.tensor.q_scale();
                    QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(
                        scale as f32,
                    ))
                }
            },
        }
    }
}

impl TensorMetadata for TchQTensor {
    fn dtype(&self) -> DType {
        DType::QFloat(self.scheme)
    }

    fn shape(&self) -> Shape {
        self.qtensor.shape()
    }
}

impl QTensorPrimitive for TchQTensor {
    fn scheme(&self) -> &QuantizationScheme {
        &self.scheme
    }
}

#[cfg(test)]
mod tests {
    use crate::LibTorch;

    use super::*;
    use burn_tensor::ops::QTensorOps;
    use burn_tensor::quantization::QuantizationParametersPrimitive;
    use burn_tensor::{Distribution, Tensor, TensorPrimitive};
    use rand::prelude::StdRng;
    use rand::SeedableRng;

    #[test]
    fn should_support_into_and_from_data_1d() {
        let data_expected = TensorData::random::<f32, _, _>(
            Shape::new([3]),
            Distribution::Default,
            &mut StdRng::from_os_rng(),
        );
        let tensor = TchTensor::from_data::<f32>(data_expected.clone(), tch::Device::Cpu);

        let data_actual =
            Tensor::<LibTorch<f32>, 1>::from_primitive(TensorPrimitive::Float(tensor)).into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_2d() {
        let data_expected = TensorData::random::<f32, _, _>(
            Shape::new([2, 3]),
            Distribution::Default,
            &mut StdRng::from_os_rng(),
        );
        let tensor = TchTensor::from_data::<f32>(data_expected.clone(), tch::Device::Cpu);

        let data_actual =
            Tensor::<LibTorch<f32>, 2>::from_primitive(TensorPrimitive::Float(tensor)).into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_not_update_inplace_after_reshape() {
        let tensor_1 = Tensor::<LibTorch<f32>, 1>::from_floats([4.0, 4.0], &Default::default());
        let tensor_2 = tensor_1.clone();

        let tensor_3 = tensor_2.reshape([1, 2]).add_scalar(2.0);

        assert_ne!(
            tensor_3.to_data().as_slice::<f32>().unwrap(),
            tensor_1.to_data().as_slice::<f32>().unwrap()
        );
    }

    #[test]
    fn should_not_update_inplace_after_slice() {
        let tensor_1 = Tensor::<LibTorch<f32>, 1>::from_floats([4.0, 4.0], &Default::default());
        let tensor_2 = tensor_1.clone();

        let tensor_3 = tensor_2.slice([0..2]).add_scalar(2.0);

        assert_ne!(
            tensor_3.to_data().as_slice::<f32>().unwrap(),
            tensor_1.to_data().as_slice::<f32>().unwrap()
        );
    }

    #[test]
    fn should_support_qtensor_strategy() {
        let tensor =
            TchTensor::from_data::<f32>(TensorData::from([-1.8, -1.0, 0.0, 0.5]), tch::Device::Cpu);
        let scheme = QuantizationScheme::PerTensorAffine(QuantizationType::QInt8);
        let qparams = QuantizationParametersPrimitive::<LibTorch<f32, i8>> {
            scale: TchTensor::from_data::<f32>(TensorData::from([0.009_019_608]), tch::Device::Cpu),
            offset: Some(TchTensor::from_data::<i8>(
                TensorData::from([72]),
                tch::Device::Cpu,
            )),
        };
        let qtensor: TchQTensor = LibTorch::quantize(tensor, &scheme, qparams);

        assert_eq!(qtensor.scheme(), &scheme);
        assert_eq!(
            qtensor.strategy(),
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.009_019_608, 72))
        );
    }
}
