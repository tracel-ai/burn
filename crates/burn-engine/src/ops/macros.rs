#[macro_export]
macro_rules! dispatch_device {
    ($device:expr, $op:expr $(, $($args:expr),* )? $(,)?) => {{
        match $device {
            #[cfg(feature = "cpu")]
            Device::Cpu(inner) => {
                type B = Cpu<f32>;
                $op(inner $(, $($args),*)?)
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(inner) => {
                type B = Cuda<f32>;
                $op(inner $(, $($args),*)?)
            }
            #[cfg(feature = "metal")]
            Device::Metal(inner) => {
                type B = Metal<f32>;
                $op(inner $(, $($args),*)?)
            }
            #[cfg(feature = "rocm")]
            Device::Rocm(inner) => {
                type B = Rocm<f32>;
                $op(inner $(, $($args),*)?)
            }
            #[cfg(feature = "vulkan")]
            Device::Vulkan(inner) => {
                type B = Vulkan<f32>;
                $op(inner $(, $($args),*)?)
            }
            #[cfg(feature = "webgpu")]
            Device::WebGpu(inner) => {
                type B = WebGpu<f32>;
                $op(inner $(, $($args),*)?)
            }
            #[cfg(feature = "ndarray")]
            Device::NdArray(inner) => {
                type B = NdArray<f32>;
                $op(inner $(, $($args),*)?)
            }
            #[cfg(feature = "tch")]
            Device::LibTorch(inner) => {
                type B = LibTorch<f32>;
                $op(inner $(, $($args),*)?)
            }
        }
    }};
}

#[macro_export]
macro_rules! backend_data {
    ($callback:ident, $($extra:tt)*) => {
        $crate::$callback! {
            $($extra)*;
            [Cpu, "cpu"] => [[Cuda, "cuda"], [Metal, "metal"], [Rocm, "rocm"], [Vulkan, "vulkan"], [WebGpu, "webgpu"], [NdArray, "ndarray"], [LibTorch, "tch"]];
            [Cuda, "cuda"] => [[Cpu, "cpu"], [Metal, "metal"], [Rocm, "rocm"], [Vulkan, "vulkan"], [WebGpu, "webgpu"], [NdArray, "ndarray"], [LibTorch, "tch"]];
            [Metal, "metal"] => [[Cpu, "cpu"], [Cuda, "cuda"], [Rocm, "rocm"], [NdArray, "ndarray"], [LibTorch, "tch"]];
            [Rocm, "rocm"] => [[Cpu, "cpu"], [Cuda, "cuda"], [Metal, "metal"], [Vulkan, "vulkan"], [WebGpu, "webgpu"], [NdArray, "ndarray"], [LibTorch, "tch"]];
            [Vulkan, "vulkan"] => [[Cpu, "cpu"], [Cuda, "cuda"], [Rocm, "rocm"], [NdArray, "ndarray"], [LibTorch, "tch"]];
            [WebGpu, "webgpu"] => [[Cpu, "cpu"], [Cuda, "cuda"], [Rocm, "rocm"], [NdArray, "ndarray"], [LibTorch, "tch"]];
            [NdArray, "ndarray"] => [[Cpu, "cpu"], [Cuda, "cuda"], [Metal, "metal"], [Rocm, "rocm"], [Vulkan, "vulkan"], [WebGpu, "webgpu"], [LibTorch, "tch"]];
            [LibTorch, "tch"] => [[Cpu, "cpu"], [Cuda, "cuda"], [Metal, "metal"], [Rocm, "rocm"], [Vulkan, "vulkan"], [WebGpu, "webgpu"], [NdArray, "ndarray"]]
        }
    };
}

#[macro_export]
macro_rules! apply_to_device {
    (
        $kind:ident, $inner_fn:ident, $tensor:expr, $device:expr, $to_device:ident, |$inner:ident, $device_ident:ident| $body:expr;
        $( [$B1:ident, $src_feature:literal] => [ $( [$B2:ident, $dst_feature:literal] ),+ ] );*
    ) => {
        match ($tensor, $device) {
            // --- Same backend to_device ---
            $(
                #[cfg(feature = $src_feature)]
                ($crate::EngineTensor::$B1(tensor), $crate::Device::$B1(d)) => {
                    $crate::EngineTensor::$B1($crate::BackendTensor::$kind(
                        $B1::<f32>::$to_device(tensor.$inner_fn(), d)
                    ))
                }
            )*

            // --- Cross backend arms ---
            // This loop generates the grid of combinations
            $(
                $(
                    #[cfg(all(feature = $src_feature, feature = $dst_feature))]
                    ($crate::EngineTensor::$B1(tensor), $crate::Device::$B2($device_ident)) => {
                        type B1 = $B1<f32>;
                        type B2 = $B2<f32>;
                        let $inner = tensor.$inner_fn();

                        $crate::EngineTensor::$B2(
                            $crate::BackendTensor::$kind($body)
                        )
                    }
                )+
            )*

            _ => todo!("Requested device combination is not supported or feature-gated off"),
        }
    };
}

#[macro_export]
macro_rules! to_device {
    ($kind:ident, $inner_fn:ident, $tensor:expr, $device:expr, $to_device:ident, |$inner:ident, $device_ident:ident| $body:expr) => {
        $crate::backend_data!(
            apply_to_device,
            $kind,
            $inner_fn,
            $tensor,
            $device,
            $to_device,
            |$inner, $device_ident| $body
        )
    };
}

// #[macro_export]
// macro_rules! to_device {
//     ($kind:ident, $inner_fn:ident, $tensor:expr, $device:expr, $to_device:ident, |$t:ident, $d:ident| $to_backend:expr) => {{
//         match ($tensor, $device) {
//             // To same backend, different device
//             #[cfg(feature = "cpu")]
//             (EngineTensor::Cpu(tensor), Device::Cpu($d)) => {
//                 $crate::EngineTensor::Cpu($crate::BackendTensor::$kind({
//                     Cpu::<f32>::$to_device(tensor.$inner_fn(), $d)
//                 }))
//             }
//             #[cfg(feature = "cuda")]
//             (EngineTensor::Cuda(tensor), Device::Cuda($d)) => $crate::EngineTensor::Cuda(
//                 $crate::BackendTensor::$kind(Cuda::<f32>::$to_device(tensor.$inner_fn(), $d)),
//             ),
//             #[cfg(feature = "metal")]
//             (EngineTensor::Metal(tensor), Device::Metal($d)) => $crate::EngineTensor::Metal(
//                 $crate::BackendTensor::$kind(Metal::<f32>::$to_device(tensor.$inner_fn(), $d)),
//             ),
//             #[cfg(feature = "rocm")]
//             (EngineTensor::Rocm(tensor), Device::Rocm($d)) => $crate::EngineTensor::Rocm(
//                 $crate::BackendTensor::$kind(Rocm::<f32>::$to_device(tensor.$inner_fn(), $d)),
//             ),
//             #[cfg(feature = "vulkan")]
//             (EngineTensor::Vulkan(tensor), Device::Vulkan($d)) => $crate::EngineTensor::Vulkan(
//                 $crate::BackendTensor::$kind(Vulkan::<f32>::$to_device(tensor.$inner_fn(), $d)),
//             ),
//             #[cfg(feature = "webgpu")]
//             (EngineTensor::WebGpu(tensor), Device::WebGpu($d)) => $crate::EngineTensor::WebGpu(
//                 $crate::BackendTensor::$kind(WebGpu::<f32>::$to_device(tensor.$inner_fn(), $d)),
//             ),
//             #[cfg(feature = "ndarray")]
//             (EngineTensor::NdArray(tensor), Device::NdArray($d)) => $crate::EngineTensor::NdArray(
//                 $crate::BackendTensor::$kind(NdArray::<f32>::$to_device(tensor.$inner_fn(), $d)),
//             ),
//             #[cfg(feature = "tch")]
//             (EngineTensor::LibTorch(tensor), Device::LibTorch($d)) => {
//                 $crate::EngineTensor::LibTorch($crate::BackendTensor::$kind(
//                     LibTorch::<f32>::$to_device(tensor.$inner_fn(), $d),
//                 ))
//             }
//             // From Cpu to other backend
//             #[cfg(all(feature = "cpu", feature = "cuda"))]
//             (EngineTensor::Cpu(tensor), Device::Cuda($d)) => {
//                 type B1 = Cpu<f32>;
//                 type B2 = Cuda<f32>;
//                 let $t = tensor.$inner_fn();
//                 $crate::EngineTensor::Cuda($crate::BackendTensor::$kind($to_backend))
//             }
//             #[cfg(all(feature = "cpu", feature = "metal"))]
//             (EngineTensor::Cpu(tensor), Device::Metal($d)) => {
//                 type B1 = Cpu<f32>;
//                 type B2 = Metal<f32>;
//                 let $t = tensor.$inner_fn();
//                 $crate::EngineTensor::Metal($crate::BackendTensor::$kind($to_backend))
//             }
//             #[cfg(all(feature = "cpu", feature = "rocm"))]
//             (EngineTensor::Cpu(tensor), Device::Rocm($d)) => {
//                 type B1 = Cpu<f32>;
//                 type B2 = Rocm<f32>;
//                 let $t = tensor.$inner_fn();
//                 $crate::EngineTensor::Rocm($crate::BackendTensor::$kind($to_backend))
//             }
//             #[cfg(all(feature = "cpu", feature = "vulkan"))]
//             (EngineTensor::Cpu(tensor), Device::Vulkan($d)) => {
//                 type B1 = Cpu<f32>;
//                 type B2 = Vulkan<f32>;
//                 let $t = tensor.$inner_fn();
//                 $crate::EngineTensor::Vulkan($crate::BackendTensor::$kind($to_backend))
//             }
//             #[cfg(all(feature = "cpu", feature = "webgpu"))]
//             (EngineTensor::Cpu(tensor), Device::WebGpu($d)) => {
//                 type B1 = Cpu<f32>;
//                 type B2 = WebGpu<f32>;
//                 let $t = tensor.$inner_fn();
//                 $crate::EngineTensor::WebGpu($crate::BackendTensor::$kind($to_backend))
//             }
//             #[cfg(all(feature = "cpu", feature = "ndarray"))]
//             (EngineTensor::Cpu(tensor), Device::NdArray($d)) => {
//                 type B1 = Cpu<f32>;
//                 type B2 = NdArray<f32>;
//                 let $t = tensor.$inner_fn();
//                 $crate::EngineTensor::NdArray($crate::BackendTensor::$kind($to_backend))
//             }
//             #[cfg(all(feature = "cpu", feature = "tch"))]
//             (EngineTensor::Cpu(tensor), Device::LibTorch($d)) => {
//                 type B1 = Cpu<f32>;
//                 type B2 = LibTorch<f32>;
//                 let $t = tensor.$inner_fn();
//                 $crate::EngineTensor::LibTorch($crate::BackendTensor::$kind($to_backend))
//             }
//             // From Cuda to other backend
//             #[cfg(all(feature = "cuda", feature = "cpu"))]
//             (EngineTensor::Cuda(tensor), Device::Cpu($d)) => {
//                 type B1 = Cuda<f32>;
//                 type B2 = Cpu<f32>;
//                 let $t = tensor.$inner_fn();
//                 $crate::EngineTensor::Cpu($crate::BackendTensor::$kind($to_backend))
//             }
//             #[cfg(all(feature = "cuda", feature = "metal"))]
//             (EngineTensor::Cuda(tensor), Device::Metal($d)) => {
//                 type B1 = Cuda<f32>;
//                 type B2 = Metal<f32>;
//                 let $t = tensor.$inner_fn();
//                 $crate::EngineTensor::Metal($crate::BackendTensor::$kind($to_backend))
//             }
//             #[cfg(all(feature = "cuda", feature = "rocm"))]
//             (EngineTensor::Cuda(tensor), Device::Rocm($d)) => {
//                 type B1 = Cuda<f32>;
//                 type B2 = Rocm<f32>;
//                 let $t = tensor.$inner_fn();
//                 $crate::EngineTensor::Rocm($crate::BackendTensor::$kind($to_backend))
//             }
//             #[cfg(all(feature = "cuda", feature = "vulkan"))]
//             (EngineTensor::Cuda(tensor), Device::Vulkan($d)) => {
//                 type B1 = Cuda<f32>;
//                 type B2 = Vulkan<f32>;
//                 let $t = tensor.$inner_fn();
//                 $crate::EngineTensor::Vulkan($crate::BackendTensor::$kind($to_backend))
//             }
//             #[cfg(all(feature = "cuda", feature = "webgpu"))]
//             (EngineTensor::Cuda(tensor), Device::WebGpu($d)) => {
//                 type B1 = Cuda<f32>;
//                 type B2 = WebGpu<f32>;
//                 let $t = tensor.$inner_fn();
//                 $crate::EngineTensor::WebGpu($crate::BackendTensor::$kind($to_backend))
//             }
//             #[cfg(all(feature = "cuda", feature = "ndarray"))]
//             (EngineTensor::Cuda(tensor), Device::NdArray($d)) => {
//                 type B1 = Cuda<f32>;
//                 type B2 = NdArray<f32>;
//                 let $t = tensor.$inner_fn();
//                 $crate::EngineTensor::NdArray($crate::BackendTensor::$kind($to_backend))
//             }
//             #[cfg(all(feature = "cuda", feature = "tch"))]
//             (EngineTensor::Cuda(tensor), Device::LibTorch($d)) => {
//                 type B1 = Cuda<f32>;
//                 type B2 = LibTorch<f32>;
//                 let $t = tensor.$inner_fn();
//                 $crate::EngineTensor::LibTorch($crate::BackendTensor::$kind($to_backend))
//             }
//             // TODO all combinations
//             _ => todo!(),
//         }
//     }};
// }

#[macro_export]
macro_rules! creation_op {
    ($kind:ident, $device:expr, $op:expr) => {{
        match $device {
            #[cfg(feature = "cpu")]
            Device::Cpu(inner) => EngineTensor::Cpu($crate::BackendTensor::$kind({
                type B = Cpu<f32>;
                $op(inner)
            })),
            #[cfg(feature = "cuda")]
            Device::Cuda(inner) => EngineTensor::Cuda($crate::BackendTensor::$kind({
                type B = Cuda<f32>;
                $op(inner)
            })),
            #[cfg(feature = "metal")]
            Device::Metal(inner) => EngineTensor::Metal($crate::BackendTensor::$kind({
                type B = Metal<f32>;
                $op(inner)
            })),
            #[cfg(feature = "rocm")]
            Device::Rocm(inner) => EngineTensor::Rocm($crate::BackendTensor::$kind({
                type B = Rocm<f32>;
                $op(inner)
            })),
            #[cfg(feature = "vulkan")]
            Device::Vulkan(inner) => EngineTensor::Vulkan($crate::BackendTensor::$kind({
                type B = Vulkan<f32>;
                $op(inner)
            })),
            #[cfg(feature = "webgpu")]
            Device::WebGpu(inner) => EngineTensor::WebGpu($crate::BackendTensor::$kind({
                type B = WebGpu<f32>;
                $op(inner)
            })),
            #[cfg(feature = "ndarray")]
            Device::NdArray(inner) => EngineTensor::NdArray($crate::BackendTensor::$kind({
                type B = NdArray<f32>;
                $op(inner)
            })),
            #[cfg(feature = "tch")]
            Device::LibTorch(inner) => EngineTensor::LibTorch($crate::BackendTensor::$kind({
                type B = LibTorch<f32>;
                $op(inner)
            })),
        }
    }};
}

#[macro_export]
macro_rules! dispatch_async_op {
    ($inner_fn:ident, $op:ident, $tensor:expr $(, $($args:expr),* )? $(,)?) => {
        match $tensor {
            #[cfg(feature = "cpu")]
            EngineTensor::Cpu(inner) => Cpu::<f32>::$op(inner.$inner_fn() $(, $($args),*)?).await,
            #[cfg(feature = "cuda")]
            EngineTensor::Cuda(inner) => Cuda::<f32>::$op(inner.$inner_fn() $(, $($args),*)?).await,
            #[cfg(feature = "metal")]
            EngineTensor::Metal(inner) => Metal::<f32>::$op(inner.$inner_fn() $(, $($args),*)?).await,
            #[cfg(feature = "rocm")]
            EngineTensor::Rocm(inner) => Rocm::<f32>::$op(inner.$inner_fn() $(, $($args),*)?).await,
            #[cfg(feature = "vulkan")]
            EngineTensor::Vulkan(inner) => Vulkan::<f32>::$op(inner.$inner_fn() $(, $($args),*)?).await,
            #[cfg(feature = "webgpu")]
            EngineTensor::WebGpu(inner) => WebGpu::<f32>::$op(inner.$inner_fn() $(, $($args),*)?).await,
            #[cfg(feature = "ndarray")]
            EngineTensor::NdArray(inner) => NdArray::<f32>::$op(inner.$inner_fn() $(, $($args),*)?).await,
            #[cfg(feature = "tch")]
            EngineTensor::LibTorch(inner) => LibTorch::<f32>::$op(inner.$inner_fn() $(, $($args),*)?).await
        }
    };
}

#[macro_export]
macro_rules! unary_op {
    ($kind:ident, $inner_fn:ident, $op:ident, $tensor:expr $(, $($args:expr),* )? $(,)?) => {
        match $tensor {
            #[cfg(feature = "cpu")]
            EngineTensor::Cpu(inner) => {
                EngineTensor::Cpu($crate::BackendTensor::$kind(Cpu::<f32>::$op(inner.$inner_fn() $(, $($args),*)?)))
            }
            #[cfg(feature = "cuda")]
            EngineTensor::Cuda(inner) => {
                EngineTensor::Cuda($crate::BackendTensor::$kind(Cuda::<f32>::$op(inner.$inner_fn() $(, $($args),*)?)))
            }
            #[cfg(feature = "metal")]
            EngineTensor::Metal(inner) => {
                EngineTensor::Metal($crate::BackendTensor::$kind(Metal::<f32>::$op(inner.$inner_fn() $(, $($args),*)?)))
            }
            #[cfg(feature = "rocm")]
            EngineTensor::Rocm(inner) => {
                EngineTensor::Rocm($crate::BackendTensor::$kind(Rocm::<f32>::$op(inner.$inner_fn() $(, $($args),*)?)))
            }
            #[cfg(feature = "vulkan")]
            EngineTensor::Vulkan(inner) => {
                EngineTensor::Vulkan($crate::BackendTensor::$kind(Vulkan::<f32>::$op(inner.$inner_fn() $(, $($args),*)?)))
            }
            #[cfg(feature = "webgpu")]
            EngineTensor::WebGpu(inner) => {
                EngineTensor::WebGpu($crate::BackendTensor::$kind(WebGpu::<f32>::$op(inner.$inner_fn() $(, $($args),*)?)))
            }
            #[cfg(feature = "ndarray")]
            EngineTensor::NdArray(inner) => {
                EngineTensor::NdArray($crate::BackendTensor::$kind(NdArray::<f32>::$op(inner.$inner_fn() $(, $($args),*)?)))
            }
            #[cfg(feature = "tch")]
            EngineTensor::LibTorch(inner) => {
                EngineTensor::LibTorch($crate::BackendTensor::$kind(LibTorch::<f32>::$op(inner.$inner_fn() $(, $($args),*)?)))
            }
        }
    };
}

/// Dispatch a float tensor creation operation.
#[macro_export]
macro_rules! create_float {
    ($device:expr, $op:expr $(,)?) => {
        $crate::creation_op!(Float, $device, $op)
    };
}

/// Dispatch an int tensor creation operation.
#[macro_export]
macro_rules! create_int {
    ($device:expr, $op:expr $(,)?) => {
        $crate::creation_op!(Int, $device, $op)
    };
}

/// Dispatch a bool tensor creation operation.
#[macro_export]
macro_rules! create_bool {
    ($device:expr, $op:expr $(,)?) => {
        $crate::creation_op!(Bool, $device, $op)
    };
}

/// Dispatch a quantized tensor creation operation.
#[macro_export]
macro_rules! create_quantized {
    ($device:expr, $op:expr $(,)?) => {
        $crate::creation_op!(Quantized, $device, $op)
    };
}

/// Dispatch a float tensor operation where the result is not wrapped.
#[macro_export]
macro_rules! dispatch_async_float {
    ($op:ident, $tensor:expr$(, $($args:expr),* )? $(,)?) => {
        $crate::dispatch_async_op!(float, $op, $tensor $(, $($args),*)?)
    };
}

/// Dispatch an int tensor operation where the result is not wrapped.
#[macro_export]
macro_rules! dispatch_async_int {
    ($op:ident, $tensor:expr$(, $($args:expr),* )? $(,)?) => {
        $crate::dispatch_async_op!(int, $op, $tensor $(, $($args),*)?)
    };
}

/// Dispatch a bool tensor operation where the result is not wrapped.
#[macro_export]
macro_rules! dispatch_async_bool {
    ($op:ident, $tensor:expr$(, $($args:expr),* )? $(,)?) => {
        $crate::dispatch_async_op!(bool, $op, $tensor $(, $($args),*)?)
    };
}

/// Dispatch a quantized tensor operation where the result is not wrapped.
#[macro_export]
macro_rules! dispatch_async_quantized {
    ($op:ident, $tensor:expr$(, $($args:expr),* )? $(,)?) => {
        $crate::dispatch_async_op!(quantized, $op, $tensor $(, $($args),*)?)
    };
}

/// Dispatch a unary float tensor operation.
#[macro_export]
macro_rules! unary_float {
    ($op:ident, $tensor:expr$(, $($args:expr),* )? $(,)?) => {
        $crate::unary_float!($op, $tensor $(, $($args),*)? => Float)
    };
    // Specify output kind
    ($op:ident, $tensor:expr$(, $($args:expr),* )? $(,)? => $out:ident) => {
        $crate::unary_op!($out, float, $op, $tensor $(, $($args),*)?)
    };
}

/// Dispatch a unary int tensor operation.
#[macro_export]
macro_rules! unary_int {
    ($op:ident, $tensor:expr$(, $($args:expr),* )? $(,)?) => {
        $crate::unary_int!($op, $tensor $(, $($args),*)? => Int)
    };
    // Specify output kind
    ($op:ident, $tensor:expr$(, $($args:expr),* )? $(,)? => $out:ident) => {
        $crate::unary_op!($out, int, $op, $tensor $(, $($args),*)?)
    };
}

/// Dispatch a unary bool tensor operation.
#[macro_export]
macro_rules! unary_bool {
    ($op:ident, $tensor:expr$(, $($args:expr),* )? $(,)?) => {
        $crate::unary_int!($op, $tensor $(, $($args),*)? => Bool)
    };
    // Specify output kind
    ($op:ident, $tensor:expr$(, $($args:expr),* )? $(,)? => $out:ident) => {
        $crate::unary_op!($out, bool, $op, $tensor $(, $($args),*)?)
    };
}

/// Dispatch a unary quantized tensor operation.
#[macro_export]
macro_rules! unary_quantized {
    ($op:ident, $tensor:expr$(, $($args:expr),* )? $(,)?) => {
        $crate::unary_quantized!($op, $tensor $(, $($args),*)? => Quantized)
    };
    // Specify output kind
    ($op:ident, $tensor:expr$(, $($args:expr),* )? $(,)? => $out:ident) => {
        $crate::unary_op!($out, quantized, $op, $tensor $(, $($args),*)?)
    };
}

/// Dispatch a binary float tensor operation.
#[macro_export]
macro_rules! binary_float {
    ($op:ident, $lhs:expr, $rhs:expr$(, $($args:expr),* )? $(,)?) => {
        $crate::binary_float!($op, $lhs, $rhs $(, $($args),*)? => Float)
    };
    // Specify output kind
    ($op:ident, $lhs:expr, $rhs:expr$(, $($args:expr),* )? $(,)? => $out:ident) => {
        $crate::multi_tensor_op!($out, float($lhs), float($rhs), |lhs, rhs| B::$op(lhs, rhs $(, $($args),*)?))
    };
}

/// Dispatch a binary int tensor operation.
#[macro_export]
macro_rules! binary_int {
    ($op:ident, $lhs:expr, $rhs:expr$(, $($args:expr),* )? $(,)?) => {
        $crate::binary_int!($op, $lhs, $rhs $(, $($args),*)? => Int)
    };
    // Specify output kind
    ($op:ident, $lhs:expr, $rhs:expr$(, $($args:expr),* )? $(,)? => $out:ident) => {
        $crate::multi_tensor_op!($out, int($lhs), int($rhs), |lhs, rhs| B::$op(lhs, rhs $(, $($args),*)?))
    };
}

/// Dispatch a binary bool tensor operation.
#[macro_export]
macro_rules! binary_bool {
    ($op:ident, $lhs:expr, $rhs:expr$(, $($args:expr),* )? $(,)?) => {
        $crate::binary_bool!($op, $lhs, $rhs $(, $($args),*)? => Bool)
    };
    // Specify output kind
    ($op:ident, $lhs:expr, $rhs:expr$(, $($args:expr),* )? $(,)? => $out:ident) => {
        $crate::multi_tensor_op!($out, bool($lhs), bool($rhs), |lhs, rhs| B::$op(lhs, rhs $(, $($args),*)?))
    };
}

/// Dispatch a binary quantized tensor operation.
#[macro_export]
macro_rules! binary_quantized {
    ($op:ident, $lhs:expr, $rhs:expr$(, $($args:expr),* )? $(,)?) => {
        $crate::binary_quantized!($op, $lhs, $rhs $(, $($args),*)? => Quantized)
    };
    // Specify output kind
    ($op:ident, $lhs:expr, $rhs:expr$(, $($args:expr),* )? $(,)? => $out:ident) => {
        $crate::multi_tensor_op!($out, quantized($lhs), quantized($rhs), |lhs, rhs| B::$op(lhs, rhs $(, $($args),*)?))
    };
}

#[macro_export]
macro_rules! multi_tensor_op {
    // -----------------------------------
    // Operations on 2 tensors
    // -----------------------------------
    // Float with float (binary_float)
    ($kind:ident, float($tensor1:expr), float($tensor2:expr), |$t1:ident, $t2:ident| $body:expr) => {
        $crate::multi_tensor_op!($kind, float, float, $tensor1, $tensor2, |$t1, $t2| $body)
    };
    // Float with int
    ($kind:ident, float($tensor1:expr), int($tensor2:expr), |$t1:ident, $t2:ident| $body:expr) => {
        $crate::multi_tensor_op!($kind, float, int, $tensor1, $tensor2, |$t1, $t2| $body)
    };
    // Float with bool
    ($kind:ident, float($tensor1:expr), bool($tensor2:expr), |$t1:ident, $t2:ident| $body:expr) => {
        $crate::multi_tensor_op!($kind, float, bool, $tensor1, $tensor2, |$t1, $t2| $body)
    };
    // Int with int (binary_int)
    ($kind:ident, int($tensor1:expr), int($tensor2:expr), |$t1:ident, $t2:ident| $body:expr) => {
        $crate::multi_tensor_op!($kind, int, int, $tensor1, $tensor2, |$t1, $t2| $body)
    };
    // Int with bool
    ($kind:ident, int($tensor1:expr), bool($tensor2:expr), |$t1:ident, $t2:ident| $body:expr) => {
        $crate::multi_tensor_op!($kind, int, bool, $tensor1, $tensor2, |$t1, $t2| $body)
    };
    // Bool with bool (binary_bool)
    ($kind:ident, bool($tensor1:expr), bool($tensor2:expr), |$t1:ident, $t2:ident| $body:expr) => {
        $crate::multi_tensor_op!($kind, bool, bool, $tensor1, $tensor2, |$t1, $t2| $body)
    };
    // Bool with int
    ($kind:ident, bool($tensor1:expr), int($tensor2:expr), |$t1:ident, $t2:ident| $body:expr) => {
        $crate::multi_tensor_op!($kind, bool, int, $tensor1, $tensor2, |$t1, $t2| $body)
    };
    ($kind:ident, $inner_fn1:ident, $inner_fn2:ident, $tensor1:expr, $tensor2:expr, |$t1:ident, $t2:ident| $body:expr) => {
        match ($tensor1, $tensor2) {
            #[cfg(feature = "cpu")]
            ($crate::EngineTensor::Cpu($t1), $crate::EngineTensor::Cpu($t2)) => {
                $crate::EngineTensor::Cpu($crate::BackendTensor::$kind({
                    type B = Cpu<f32>;
                    let $t1 = $t1.$inner_fn1();
                    let $t2 = $t2.$inner_fn2();
                    $body
                }))
            }
            #[cfg(feature = "cuda")]
            ($crate::EngineTensor::Cuda($t1), $crate::EngineTensor::Cuda($t2)) => {
                $crate::EngineTensor::Cuda($crate::BackendTensor::$kind({
                    type B = Cuda<f32>;
                    let $t1 = $t1.$inner_fn1();
                    let $t2 = $t2.$inner_fn2();
                    $body
                }))
            }
            #[cfg(feature = "metal")]
            ($crate::EngineTensor::Metal($t1), $crate::EngineTensor::Metal($t2)) => {
                $crate::EngineTensor::Metal($crate::BackendTensor::$kind({
                    type B = Metal<f32>;
                    let $t1 = $t1.$inner_fn1();
                    let $t2 = $t2.$inner_fn2();
                    $body
                }))
            }
            #[cfg(feature = "rocm")]
            ($crate::EngineTensor::Rocm($t1), $crate::EngineTensor::Rocm($t2)) => {
                $crate::EngineTensor::Rocm($crate::BackendTensor::$kind({
                    type B = Rocm<f32>;
                    let $t1 = $t1.$inner_fn1();
                    let $t2 = $t2.$inner_fn2();
                    $body
                }))
            }
            #[cfg(feature = "vulkan")]
            ($crate::EngineTensor::Vulkan($t1), $crate::EngineTensor::Vulkan($t2)) => {
                $crate::EngineTensor::Vulkan($crate::BackendTensor::$kind({
                    type B = Vulkan<f32>;
                    let $t1 = $t1.$inner_fn1();
                    let $t2 = $t2.$inner_fn2();
                    $body
                }))
            }
            #[cfg(feature = "webgpu")]
            ($crate::EngineTensor::WebGpu($t1), $crate::EngineTensor::WebGpu($t2)) => {
                $crate::EngineTensor::WebGpu($crate::BackendTensor::$kind({
                    type B = WebGpu<f32>;
                    let $t1 = $t1.$inner_fn1();
                    let $t2 = $t2.$inner_fn2();
                    $body
                }))
            }
            #[cfg(feature = "ndarray")]
            ($crate::EngineTensor::NdArray($t1), $crate::EngineTensor::NdArray($t2)) => {
                $crate::EngineTensor::NdArray($crate::BackendTensor::$kind({
                    type B = NdArray<f32>;
                    let $t1 = $t1.$inner_fn1();
                    let $t2 = $t2.$inner_fn2();
                    $body
                }))
            }
            #[cfg(feature = "tch")]
            ($crate::EngineTensor::LibTorch($t1), $crate::EngineTensor::LibTorch($t2)) => {
                $crate::EngineTensor::LibTorch($crate::BackendTensor::$kind({
                    type B = LibTorch<f32>;
                    let $t1 = $t1.$inner_fn1();
                    let $t2 = $t2.$inner_fn2();
                    $body
                }))
            }
            ($t1, $t2) => {
                panic!(
                    "The provided tensors are not on the same device. Got devices {:?} and {:?}.", $t1, $t2
                );
            }
        }
    };

    // -----------------------------------
    // Operations on 3 tensors
    // -----------------------------------
    // Float - int - float
    ($kind:ident, float($tensor1:expr), int($tensor2:expr), float($tensor3:expr), |$t1:ident, $t2:ident, $t3:ident| $body:expr) => {
        $crate::multi_tensor_op!($kind, float, int, float, $tensor1, $tensor2, $tensor3, |$t1, $t2, $t3| $body)
    };
    // Float - bool - float
    ($kind:ident, float($tensor1:expr), bool($tensor2:expr), float($tensor3:expr), |$t1:ident, $t2:ident, $t3:ident| $body:expr) => {
        $crate::multi_tensor_op!($kind, float, bool, float, $tensor1, $tensor2, $tensor3, |$t1, $t2, $t3| $body)
    };
    // Int - int - int
    ($kind:ident, int($tensor1:expr), int($tensor2:expr), int($tensor3:expr), |$t1:ident, $t2:ident, $t3:ident| $body:expr) => {
        $crate::multi_tensor_op!($kind, int, int, int, $tensor1, $tensor2, $tensor3, |$t1, $t2, $t3| $body)
    };
    // Int - bool - int
    ($kind:ident, int($tensor1:expr), bool($tensor2:expr), int($tensor3:expr), |$t1:ident, $t2:ident, $t3:ident| $body:expr) => {
        $crate::multi_tensor_op!($kind, int, bool, int, $tensor1, $tensor2, $tensor3, |$t1, $t2, $t3| $body)
    };
    // Bool - bool - bool
    ($kind:ident, bool($tensor1:expr), bool($tensor2:expr), bool($tensor3:expr), |$t1:ident, $t2:ident, $t3:ident| $body:expr) => {
        $crate::multi_tensor_op!($kind, bool, bool, bool, $tensor1, $tensor2, $tensor3, |$t1, $t2, $t3| $body)
    };
    // Bool - int - bool
    ($kind:ident, bool($tensor1:expr), int($tensor2:expr), bool($tensor3:expr), |$t1:ident, $t2:ident, $t3:ident| $body:expr) => {
        $crate::multi_tensor_op!($kind, bool, int, bool, $tensor1, $tensor2, $tensor3, |$t1, $t2, $t3| $body)
    };

    ($kind:ident, $inner_fn1:ident, $inner_fn2:ident, $inner_fn3:ident, $tensor1:expr, $tensor2:expr, $tensor3:expr, |$t1:ident, $t2:ident, $t3:ident| $body:expr) => {
        match ($tensor1, $tensor2, $tensor3) {
            #[cfg(feature = "cpu")]
            ($crate::EngineTensor::Cpu($t1), $crate::EngineTensor::Cpu($t2), $crate::EngineTensor::Cpu($t3))  => {
                $crate::EngineTensor::Cpu($crate::BackendTensor::$kind({
                    type B = Cpu<f32>;
                    let $t1 = $t1.$inner_fn1();
                    let $t2 = $t2.$inner_fn2();
                    let $t3 = $t3.$inner_fn3();
                    $body
                }))
            }
            #[cfg(feature = "cuda")]
            ($crate::EngineTensor::Cuda($t1), $crate::EngineTensor::Cuda($t2), $crate::EngineTensor::Cuda($t3)) => {
                $crate::EngineTensor::Cuda($crate::BackendTensor::$kind({
                    type B = Cuda<f32>;
                    let $t1 = $t1.$inner_fn1();
                    let $t2 = $t2.$inner_fn2();
                    let $t3 = $t3.$inner_fn3();
                    $body
                }))
            }
            #[cfg(feature = "metal")]
            ($crate::EngineTensor::Metal($t1), $crate::EngineTensor::Metal($t2), $crate::EngineTensor::Metal($t3)) => {
                $crate::EngineTensor::Metal($crate::BackendTensor::$kind({
                    type B = Metal<f32>;
                    let $t1 = $t1.$inner_fn1();
                    let $t2 = $t2.$inner_fn2();
                    let $t3 = $t3.$inner_fn3();
                    $body
                }))
            }
            #[cfg(feature = "rocm")]
            ($crate::EngineTensor::Rocm($t1), $crate::EngineTensor::Rocm($t2), $crate::EngineTensor::Rocm($t3)) => {
                $crate::EngineTensor::Rocm($crate::BackendTensor::$kind({
                    type B = Rocm<f32>;
                    let $t1 = $t1.$inner_fn1();
                    let $t2 = $t2.$inner_fn2();
                    let $t3 = $t3.$inner_fn3();
                    $body
                }))
            }
            #[cfg(feature = "vulkan")]
            ($crate::EngineTensor::Vulkan($t1), $crate::EngineTensor::Vulkan($t2), $crate::EngineTensor::Vulkan($t3)) => {
                $crate::EngineTensor::Vulkan($crate::BackendTensor::$kind({
                    type B = Vulkan<f32>;
                    let $t1 = $t1.$inner_fn1();
                    let $t2 = $t2.$inner_fn2();
                    let $t3 = $t3.$inner_fn3();
                    $body
                }))
            }
            #[cfg(feature = "webgpu")]
            ($crate::EngineTensor::WebGpu($t1), $crate::EngineTensor::WebGpu($t2), $crate::EngineTensor::WebGpu($t3)) => {
                $crate::EngineTensor::WebGpu($crate::BackendTensor::$kind({
                    type B = WebGpu<f32>;
                    let $t1 = $t1.$inner_fn1();
                    let $t2 = $t2.$inner_fn2();
                    let $t3 = $t3.$inner_fn3();
                    $body
                }))
            }
            #[cfg(feature = "ndarray")]
            ($crate::EngineTensor::NdArray($t1), $crate::EngineTensor::NdArray($t2), $crate::EngineTensor::NdArray($t3)) => {
                $crate::EngineTensor::NdArray($crate::BackendTensor::$kind({
                    type B = NdArray<f32>;
                    let $t1 = $t1.$inner_fn1();
                    let $t2 = $t2.$inner_fn2();
                    let $t3 = $t3.$inner_fn3();
                    $body
                }))
            }
            #[cfg(feature = "tch")]
            ($crate::EngineTensor::LibTorch($t1), $crate::EngineTensor::LibTorch($t2), $crate::EngineTensor::LibTorch($t3)) => {
                $crate::EngineTensor::LibTorch($crate::BackendTensor::$kind({
                    type B = LibTorch<f32>;
                    let $t1 = $t1.$inner_fn1();
                    let $t2 = $t2.$inner_fn2();
                    let $t3 = $t3.$inner_fn3();
                    $body
                }))
            }
            ($t1, $t2, $t3) => {
                panic!(
                    "The provided tensors are not on the same device. Got devices {:?}, {:?} and {:?}.", $t1, $t2, $t3
                );
            }
        }
    };
}

// A little verbose but we have to capture optional inputs and outputs...
#[macro_export]
macro_rules! module_op {
    // Pattern without int() and without optional outputs (single or multiple required outputs)
    (float($($x:ident),+), opt($($opt:ident),*) => ($($out:ident),+) $body:expr) => {{
        module_op!(float($($x),+), int(), opt($($opt),*) => ($($out),+) opt() $body)
    }};

    // Pattern without int()
    (float($($x:ident),+), opt($($opt:ident),*) => ($($out:ident),+) opt($($opt_out:ident),*) $body:expr) => {{
        module_op!(float($($x),+), int(), opt($($opt),*) => ($($out),+) opt($($opt_out),*) $body)
    }};


    (float($($x:ident),+), int($($i:ident),*), opt($($opt:ident),*) => ($($out:ident),+) opt($($opt_out:ident),*) $body:expr) => {{
        #[allow(unused_parens, unreachable_patterns)]
        match ($($x),+ $(,$i),*) {
            #[cfg(feature = "cpu")]
            ($($crate::EngineTensor::Cpu($x)),+ $(,$crate::EngineTensor::Cpu($i)),*) => {{
                type B = Cpu<f32>;
                // Unwrap the inner tensors
                $(let $x = $x.float();)+
                $(let $i = $i.float();)*
                $(let $opt = $opt.map(|o| match o { $crate::EngineTensor::Cpu(val) => val.float(), _ => panic!("Optional tensor is not on the same device.") });)*

                let ($($out),+, $($opt_out),*) = $body;

                module_op!(@wrap_outputs Cpu, ($($out),+) opt($($opt_out),*))
            }},

            #[cfg(feature = "cuda")]
            ($($crate::EngineTensor::Cuda($x)),+ $(,$crate::EngineTensor::Cuda($i)),*) => {{
                type B = Cuda<f32>;
                $(let $x = $x.float();)+
                $(let $i = $i.float();)*
                $(let $opt = $opt.map(|o| match o { $crate::EngineTensor::Cuda(val) => val.float(), _ => panic!("Optional tensor is not on the same device.") });)*

                let ($($out),+, $($opt_out),*) = $body;

                module_op!(@wrap_outputs Cuda, ($($out),+) opt($($opt_out),*))
            }},

            #[cfg(feature = "metal")]
            ($($crate::EngineTensor::Metal($x)),+ $(,$crate::EngineTensor::Metal($i)),*) => {{
                type B = Metal<f32>;
                $(let $x = $x.float();)+
                $(let $i = $i.float();)*
                $(let $opt = $opt.map(|o| match o { $crate::EngineTensor::Metal(val) => val.float(), _ => panic!("Optional tensor is not on the same device.") });)*

                let ($($out),+, $($opt_out),*) = $body;

                module_op!(@wrap_outputs Metal, ($($out),+) opt($($opt_out),*))
            }},

            #[cfg(feature = "rocm")]
            ($($crate::EngineTensor::Rocm($x)),+ $(,$crate::EngineTensor::Rocm($i)),*) => {{
                type B = Rocm<f32>;
                $(let $x = $x.float();)+
                $(let $i = $i.float();)*
                $(let $opt = $opt.map(|o| match o { $crate::EngineTensor::Rocm(val) => val.float(), _ => panic!("Optional tensor is not on the same device.") });)*

                let ($($out),+, $($opt_out),*) = $body;

                module_op!(@wrap_outputs Rocm, ($($out),+) opt($($opt_out),*))
            }},

            #[cfg(feature = "vulkan")]
            ($($crate::EngineTensor::Vulkan($x)),+ $(,$crate::EngineTensor::Vulkan($i)),*) => {{
                type B = Vulkan<f32>;
                $(let $x = $x.float();)+
                $(let $i = $i.float();)*
                $(let $opt = $opt.map(|o| match o { $crate::EngineTensor::Vulkan(val) => val.float(), _ => panic!("Optional tensor is not on the same device.") });)*

                let ($($out),+, $($opt_out),*) = $body;

                module_op!(@wrap_outputs Vulkan, ($($out),+) opt($($opt_out),*))
            }},

            #[cfg(feature = "webgpu")]
            ($($crate::EngineTensor::WebGpu($x)),+ $(,$crate::EngineTensor::WebGpu($i)),*) => {{
                type B = WebGpu<f32>;
                $(let $x = $x.float();)+
                $(let $i = $i.float();)*
                $(let $opt =  $opt.map(|o| match o { $crate::EngineTensor::WebGpu(val) => val.float(), _ => panic!("Optional tensor is not on the same device.") });)*

                let ($($out),+, $($opt_out),*) = $body;

                module_op!(@wrap_outputs WebGpu, ($($out),+) opt($($opt_out),*))
            }},

            #[cfg(feature = "ndarray")]
            ($($crate::EngineTensor::NdArray($x)),+ $(,$crate::EngineTensor::NdArray($i)),*) => {{
                type B = NdArray<f32>;
                $(let $x = $x.float();)+
                $(let $i = $i.float();)*
                $(let $opt = $opt.map(|o| match o { $crate::EngineTensor::NdArray(val) => val.float(), _ => panic!("Optional tensor is not on the same device.") });)*

                let ($($out),+, $($opt_out),*) = $body;

                module_op!(@wrap_outputs NdArray, ($($out),+) opt($($opt_out),*))
            }},

            #[cfg(feature = "tch")]
            ($($crate::EngineTensor::LibTorch($x)),+ $(,$crate::EngineTensor::LibTorch($i)),*) => {{
                type B = LibTorch<f32>;
                $(let $x = $x.float();)+
                $(let $i = $i.float();)*
                $(let $opt = $opt.map(|o| match o { $crate::EngineTensor::LibTorch(val) => val.float(), _ => panic!("Optional tensor is not on the same device.") });)*

                let ($($out),+, $($opt_out),*) = $body;

                module_op!(@wrap_outputs LibTorch, ($($out),+) opt($($opt_out),*))
            }},

            _ => panic!("The provided tensors are not on the same device."),
        }
    }};

    // Wrap outputs: required outputs + optional outputs
    (@wrap_outputs $backend:ident, ($($out_id:ident),+) opt($($opt_id:ident),*)) => {
        (
            $($crate::EngineTensor::$backend($crate::as_float($out_id))),+,
            $($opt_id.map(|t| $crate::EngineTensor::$backend($crate::as_float(t)))),*
        )
    };
}
