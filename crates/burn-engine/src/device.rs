// TODO: for engine Device enum
// /// High-level device type used by the tensor API.
// ///
// /// This type wraps the backend-specific [device](Backend::Device) and handles default
// /// floating-point and integer data types used for tensor creation. These defaults can be
// /// overriden at the call site using [`TensorCreationOptions`](crate::TensorCreationOptions).
// ///
// /// The backend device itself is accessed internally by tensor creation operations and is
// /// is not exposed publicly.
// #[derive(Debug, Default, Clone)]
// pub struct Device<B: Backend> {
//     /// Device type used by the backend.
//     inner: B::Device,
// }

// impl<B: Backend> Device<B> {
//     pub fn new(device: B::Device) -> Self {
//         Self { inner: device }
//     }
//     /// Returns the default floating-point data type used for tensor creation.
//     pub fn default_float_dtype(&self) -> FloatDType {
//         DevicePolicyRegistry::get(&self.inner).float_dtype()
//     }

//     /// Returns the default integer data type used for tensor creation.
//     pub fn default_int_dtype(&self) -> IntDType {
//         DevicePolicyRegistry::get(&self.inner).int_dtype()
//     }

//     /// Sets the default floating-point data type.
//     pub fn set_default_float_dtype(&mut self, dtype: impl Into<FloatDType>) {
//         DevicePolicyRegistry::update(&self.inner, |p| {
//             p.set_float_dtype(dtype);
//         });
//     }

//     /// Sets the default integer data type.
//     pub fn set_default_int_dtype(&mut self, dtype: impl Into<IntDType>) {
//         DevicePolicyRegistry::update(&self.inner, |p| {
//             p.set_int_dtype(dtype);
//         });
//     }

//     /// Returns the backend-specific device.
//     pub(crate) fn backend(&self) -> &B::Device {
//         &self.inner
//     }
// }

// impl<B: Backend<Device = D>, D: DeviceOps> From<D> for Device<B> {
//     fn from(value: D) -> Self {
//         Self::new(value)
//     }
// }

// previous macro
#[macro_export]
macro_rules! binary_op {
    ($kind:ident, $inner_fn:ident, $op:ident, $lhs:expr, $rhs:expr $(, $($args:expr),* )? $(,)?) => {
        match ($lhs, $rhs) {
            #[cfg(feature = "cpu")]
            ($crate::EngineTensor::Cpu(lhs), $crate::EngineTensor::Cpu(rhs)) => {
                $crate::EngineTensor::Cpu($crate::BackendTensor::$kind(
                    Cpu::<f32>::$op(lhs.$inner_fn(), rhs.$inner_fn() $(, $($args),*)?)
                ))
            }
            #[cfg(feature = "cuda")]
            ($crate::EngineTensor::Cuda(lhs), $crate::EngineTensor::Cuda(rhs)) => {
                $crate::EngineTensor::Cuda($crate::BackendTensor::$kind(
                    Cuda::<f32>::$op(lhs.$inner_fn(), rhs.$inner_fn() $(, $($args),*)?)
                ))
            }
            #[cfg(feature = "metal")]
            ($crate::EngineTensor::Metal(lhs), $crate::EngineTensor::Metal(rhs)) => {
                $crate::EngineTensor::Metal($crate::BackendTensor::$kind(
                    Metal::<f32>::$op(lhs.$inner_fn(), rhs.$inner_fn() $(, $($args),*)?)
                ))
            }
            #[cfg(feature = "rocm")]
            ($crate::EngineTensor::Rocm(lhs), $crate::EngineTensor::Rocm(rhs)) => {
                $crate::EngineTensor::Rocm($crate::BackendTensor::$kind(
                    Rocm::<f32>::$op(lhs.$inner_fn(), rhs.$inner_fn() $(, $($args),*)?)
                ))
            }
            #[cfg(feature = "vulkan")]
            ($crate::EngineTensor::Vulkan(lhs), $crate::EngineTensor::Vulkan(rhs)) => {
                $crate::EngineTensor::Vulkan($crate::BackendTensor::$kind(
                    Vulkan::<f32>::$op(lhs.$inner_fn(), rhs.$inner_fn() $(, $($args),*)?)
                ))
            }
            #[cfg(feature = "webgpu")]
            ($crate::EngineTensor::WebGpu(lhs), $crate::EngineTensor::WebGpu(rhs)) => {
                $crate::EngineTensor::WebGpu($crate::BackendTensor::$kind(
                    WebGpu::<f32>::$op(lhs.$inner_fn(), rhs.$inner_fn() $(, $($args),*)?)
                ))
            }
            #[cfg(feature = "ndarray")]
            ($crate::EngineTensor::NdArray(lhs), $crate::EngineTensor::NdArray(rhs)) => {
                $crate::EngineTensor::NdArray($crate::BackendTensor::$kind(
                    NdArray::<f32>::$op(lhs.$inner_fn(), rhs.$inner_fn() $(, $($args),*)?)
                ))
            }
            #[cfg(feature = "tch")]
            ($crate::EngineTensor::LibTorch(lhs), $crate::EngineTensor::LibTorch(rhs)) => {
                $crate::EngineTensor::LibTorch($crate::BackendTensor::$kind(
                    LibTorch::<f32>::$op(lhs.$inner_fn(), rhs.$inner_fn() $(, $($args),*)?)
                ))
            }
            (lhs, rhs) => {
                panic!(
                    "Cross-device operation attempted between {:?} and {:?}. 
                     Tensors must be moved to the same device before operations.",
                    lhs, rhs
                );
            }
        }
    };
}

/// Dispatch a binary int tensor operation.
#[macro_export]
macro_rules! binary_int {
    ($op:ident, $lhs:expr, $rhs:expr$(, $($args:expr),* )? $(,)? => $out:ident) => {
        $crate::binary_int!($op, $lhs, $rhs $(, $($args),*)? => Int)
    };
    // Specify output kind
    ($op:ident, $lhs:expr, $rhs:expr$(, $($args:expr),* )? $(,)? => $out:ident) => {
        $crate::binary_op!($out, int, $op, $lhs, $rhs $(, $($args),*)?)
    };
}
