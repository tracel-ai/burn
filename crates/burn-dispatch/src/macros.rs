/// Supplies a list of all supported backends and their corresponding feature flags
/// to a callback macro. This centralizes the backend registry.
macro_rules! backend_list {
    ($callback:ident, $($extra:tt)*) => {
        $callback! {
            $($extra)*;
            [Cpu, feature = "cpu"],
            [Cuda, feature = "cuda"],
            [Metal, wgpu_metal],
            [Rocm, feature = "rocm"],
            [Vulkan, wgpu_vulkan],
            [WebGpu, wgpu_webgpu],
            [NdArray, feature = "ndarray"],
            [LibTorch, feature = "tch"]
        }
    };
}

/// Supplies a matrix of cross-backend combinations. Used for operations where the source and destination backends may differ.
macro_rules! backend_matrix {
    ($callback:ident, $($extra:tt)*) => {
        $callback! {
            $($extra)*;
            [Cpu, feature = "cpu"] => [[Cuda, feature = "cuda"], [Metal, wgpu_metal], [Rocm, feature = "rocm"], [Vulkan, wgpu_vulkan], [WebGpu, wgpu_webgpu], [NdArray, feature = "ndarray"], [LibTorch, feature = "tch"]];
            [Cuda, feature = "cuda"] => [[Cpu, feature = "cpu"], [Metal, wgpu_metal], [Rocm, feature = "rocm"], [Vulkan, wgpu_vulkan], [WebGpu, wgpu_webgpu], [NdArray, feature = "ndarray"], [LibTorch, feature = "tch"]];
            [Metal, wgpu_metal] => [[Cpu, feature = "cpu"], [Cuda, feature = "cuda"], [Rocm, feature = "rocm"], [NdArray, feature = "ndarray"], [LibTorch, feature = "tch"]];
            [Rocm, feature = "rocm"] => [[Cpu, feature = "cpu"], [Cuda, feature = "cuda"], [Metal, wgpu_metal], [Vulkan, wgpu_vulkan], [WebGpu, wgpu_webgpu], [NdArray, feature = "ndarray"], [LibTorch, feature = "tch"]];
            [Vulkan, wgpu_vulkan] => [[Cpu, feature = "cpu"], [Cuda, feature = "cuda"], [Rocm, feature = "rocm"], [NdArray, feature = "ndarray"], [LibTorch, feature = "tch"]];
            [WebGpu, wgpu_webgpu] => [[Cpu, feature = "cpu"], [Cuda, feature = "cuda"], [Rocm, feature = "rocm"], [NdArray, feature = "ndarray"], [LibTorch, feature = "tch"]];
            [NdArray, feature = "ndarray"] => [[Cpu, feature = "cpu"], [Cuda, feature = "cuda"], [Metal, wgpu_metal], [Rocm, feature = "rocm"], [Vulkan, wgpu_vulkan], [WebGpu, wgpu_webgpu], [LibTorch, feature = "tch"]];
            [LibTorch, feature = "tch"] => [[Cpu, feature = "cpu"], [Cuda, feature = "cuda"], [Metal, wgpu_metal], [Rocm, feature = "rocm"], [Vulkan, wgpu_vulkan], [WebGpu, wgpu_webgpu], [NdArray, feature = "ndarray"]]
        }
    };
}

/// Match arm generator for `dispatch_device`.
/// Maps each backend variant to a block where the specific backend type is bound to `B`.
macro_rules! dispatch_device_arms {
    (
        $device:expr,
        |$inner:ident| $body:expr;
        $([$Backend:ident, $cfg:meta]),*
    ) => {
        match $device {
            // Autodiff arm first
            #[cfg(feature = "autodiff")]
            $crate::Device::Autodiff(inner) => {
                // Recursively dispatch on inner
                dispatch_device_arms!(
                    @autodiff
                    &**inner,
                    |$inner| $body;
                    $([$Backend, $cfg]),*
                )
            },
            $(
                #[cfg($cfg)]
                $crate::Device::$Backend($inner) => {
                    type B = $Backend<f32>;
                    $body
                }
            )*
        }
    };
    (
        @autodiff
        $device:expr,
        |$inner:ident| $body:expr;
        $([$Backend:ident, $cfg:meta]),*
    ) => {
        match $device {
            $(
                #[cfg($cfg)]
                $crate::Device::$Backend($inner) => {
                    type B = Autodiff<$Backend<f32>>;
                    $body
                }
            )*
            $crate::Device::Autodiff(_) => panic!("Autodiff should not wrap an autodiff device.")
        }
    };
}

/// Dispatches an operation body based on the provided device.
macro_rules! dispatch_device {
    ($device:expr, |$inner:ident| $body:expr) => {
        backend_list!(dispatch_device_arms, $device, |$inner| $body)
    };
}

/// Match arm generator for `to_device`.
/// Handles the logic for same-backend transfers (fast path) and cross-backend
/// transfers by generating a grid of all device combinations provided via `backend_matrix`.
macro_rules! to_device_arms {
    (
        $kind:ident, $inner_fn:ident, $tensor:expr, $device:expr, $to_device:ident, |$inner:ident, $device_ident:ident| $body:expr;
        $( [$B1:ident, $src_cfg:meta] => [ $( [$B2:ident, $dst_cfg:meta] ),+ ] );*
    ) => {
        match ($tensor, $device) {
            // --- Same backend to_device ---
            $(
                #[cfg($src_cfg)]
                ($crate::DispatchTensor::$B1(tensor), $crate::Device::$B1(d)) => {
                    $crate::DispatchTensor::$B1($crate::BackendTensor::$kind(
                        $B1::<f32>::$to_device(tensor.$inner_fn(), d)
                    ))
                }
            )*

            // --- Cross backend arms ---
            // This loop generates the grid of combinations
            $(
                $(
                    #[cfg(all($src_cfg, $dst_cfg))]
                    ($crate::DispatchTensor::$B1(tensor), $crate::Device::$B2($device_ident)) => {
                        type B1 = $B1<f32>;
                        type B2 = $B2<f32>;
                        let $inner = tensor.$inner_fn();

                        $crate::DispatchTensor::$B2(
                            $crate::BackendTensor::$kind($body)
                        )
                    }
                )+
            )*
            #[cfg(feature = "autodiff")]
            (_, $crate::Device::Autodiff(_)) | ($crate::DispatchTensor::Autodiff(_), _) => panic!("Operation not marked for autodiff.")
        }
    };
}

/// Handles tensor movement between devices, supporting both same-backend transfers
/// and cross-backend dispatches.
macro_rules! to_device {
    ($kind:ident, $inner_fn:ident, $tensor:expr, $device:expr, $to_device:ident, |$inner:ident, $device_ident:ident| $body:expr) => {
        backend_matrix!(
            to_device_arms,
            $kind,
            $inner_fn,
            $tensor,
            $device,
            $to_device,
            |$inner, $device_ident| $body
        )
    };
}

/// Match arm generator for `float_to_device`.
///
/// Similar to `to_device_arms`, but float tensors are checked for autodiff support.
macro_rules! float_to_device_arms {
    (
        $tensor:expr, $device:expr, $to_device:ident, |$inner:ident, $device_ident:ident| $body:expr;
        $( [$B1:ident, $src_cfg:meta] => [ $( [$B2:ident, $dst_cfg:meta] ),+ ] );*
    ) => {
        match ($tensor, $device) {
            #[cfg(feature = "autodiff")]
            ($crate::DispatchTensor::Autodiff(tensor), $crate::Device::Autodiff(device)) => {
                float_to_device_arms!(
                    @autodiff
                    *tensor, &**device, $to_device;
                    $([$B1, $src_cfg]);*
                )

            }
            // --- Same backend to_device ---
            $(
                #[cfg($src_cfg)]
                ($crate::DispatchTensor::$B1(tensor), $crate::Device::$B1(d)) => {
                    $crate::DispatchTensor::$B1($crate::BackendTensor::Float(
                        $B1::<f32>::$to_device(tensor.float(), d)
                    ))
                }
            )*

            // --- Cross backend arms ---
            // This loop generates the grid of combinations
            $(
                $(
                    #[cfg(all($src_cfg, $dst_cfg))]
                    ($crate::DispatchTensor::$B1(tensor), $crate::Device::$B2($device_ident)) => {
                        type B1 = $B1<f32>;
                        type B2 = $B2<f32>;
                        let $inner = tensor.float();

                        $crate::DispatchTensor::$B2(
                            $crate::BackendTensor::Float($body)
                        )
                    }
                )+
            )*
            // TODO: maybe?
            #[cfg(feature = "autodiff")]
            ($crate::DispatchTensor::Autodiff(_), _) | (_, $crate::Device::Autodiff(_)) => panic!("Cannot move between autodiff and non-autodiff instances.")
        }
    };

    // Autodiff(DispatchTensor)
    (
        @autodiff
        $tensor:expr, $device:expr, $to_device:ident;
        $( [$B1:ident, $src_cfg:meta] );*
    ) => {{
        match ($tensor, $device) {
            // --- Same backend to_device ---
            $(
                #[cfg($src_cfg)]
                ($crate::DispatchTensor::$B1(tensor), $crate::Device::$B1(d)) => {
                    $crate::DispatchTensor::Autodiff(Box::new($crate::DispatchTensor::$B1($crate::BackendTensor::Autodiff(
                        Autodiff::<$B1<f32>>::$to_device(tensor.autodiff(), d)
                    ))))
                }
            )*
            (_, _) => unimplemented!("Autodiff tensor cannot be moved between backends.")
        }
    }};
}

/// Handles float tensor movement between devices (that might support autodiff).
macro_rules! float_to_device {
    ($kind:ident, $inner_fn:ident, $tensor:expr, $device:expr, $to_device:ident, |$inner:ident, $device_ident:ident| $body:expr) => {
        backend_matrix!(
            float_to_device_arms,
            $tensor,
            $device,
            $to_device,
            |$inner, $device_ident| $body
        )
    };
}

/// Dispatches a tensor creation operation (e.g., zeros, ones) to the correct backend
/// based on the provided device.
macro_rules! creation_op {
    ($kind:ident, $device:expr, |$inner:ident| $body:expr) => {
        backend_list!(creation_op_arms, $kind, $device, |$inner| $body)
    };
}

/// Match arm generator for `creation_float`.
///
/// Similar to `creation_op_arms`, but float tensors are checked for autodiff support.
macro_rules! creation_op_arms {
    (
        $kind:ident,
        $device:expr,
        |$inner:ident| $body:expr;
        $([$Backend:ident, $cfg:meta]),*
    ) => {{
        match $device {
            // Autodiff arm first
            #[cfg(feature = "autodiff")]
            $crate::Device::Autodiff(inner) => {
                // Recursively dispatch on inner
                creation_op_arms!(
                    @autodiff
                    $kind,
                    &**inner,
                    |$inner| $body;
                    $([$Backend, $cfg]),*
                )
            },
            $(
                #[cfg($cfg)]
                $crate::Device::$Backend($inner) => {
                    type B = $Backend<f32>;
                    $crate::DispatchTensor::$Backend(
                        $crate::BackendTensor::$kind($body)
                    )
                }
            )*
        }
    }};

    (
        @autodiff
        $kind:ident,
        $device:expr,
        |$inner:ident| $body:expr;
        $([$Backend:ident, $cfg:meta]),*
    ) => {{
        match $device {
            $(
                #[cfg($cfg)]
                $crate::Device::$Backend($inner) => {
                    type B = Autodiff<$Backend<f32>>;
                    wrap_float!(
                        @wrap_autodiff
                        $kind,
                        $Backend,
                        { $body }
                    )
                }
            )*
            $crate::Device::Autodiff(_) => panic!("Autodiff should not wrap an autodiff device.")
        }
    }};
}

/// Dispatches a float tensor creation operation (that might support autodiff).
// macro_rules! creation_float {
//     ($kind:ident, $device:expr, |$inner:ident| $body:expr) => {
//         backend_list!(creation_float_arms, $kind, $device, |$inner| $body)
//     };
// }

/// Wrap the result in the backend tensor kind, handling float -> autodiff.
macro_rules! wrap_float {
    (
        @wrap_autodiff Float,
        $Backend:ident,
        $expr:expr
    ) => {
        $crate::DispatchTensor::Autodiff(Box::new($crate::DispatchTensor::$Backend(
            $crate::BackendTensor::Autodiff($expr),
        )))
    };

    (
        @wrap_autodiff $other:ident,
        $Backend:ident,
        $expr:expr
    ) => {
        $crate::DispatchTensor::$Backend($crate::BackendTensor::$other($expr))
    };
}

/// Match arm generator for `unary_op`.
/// Unwraps the inner tensor primitive (e.g., `inner.float()`) and provides the backend type `B`
/// for the operation.
///
/// When the return kind is provided, the result is wrapped in the corresponding `DispatchTensor` variant.
macro_rules! unary_op_arms {
    (
        $kind:ident,
        $inner_kind:ident,
        $tensor:expr,
        |$inner:ident| $body:expr;
        $([$Backend:ident, $cfg:meta]),*
    ) => {{
        match $tensor {
            $(
                #[cfg($cfg)]
                $crate::DispatchTensor::$Backend($inner) => {
                    type B = $Backend<f32>;
                    let $inner = $inner.$inner_kind();
                    $crate::DispatchTensor::$Backend($crate::BackendTensor::$kind($body))
                }
            )*
            #[cfg(feature = "autodiff")]
            $crate::DispatchTensor::Autodiff(_) => panic!("Operation not marked for autodiff.")
        }
    }};

    // Operations that do not return a tensor kind
    (
        $inner_kind:ident,
        $tensor:expr,
        |$inner:ident| $body:expr;
        $([$Backend:ident, $cfg:meta]),*
    ) => {{
        match $tensor {
            $(
                #[cfg($cfg)]
                $crate::DispatchTensor::$Backend($inner) => {
                    type B = $Backend<f32>;
                    let $inner = $inner.$inner_kind();
                    $body
                }
            )*
            #[cfg(feature = "autodiff")]
            $crate::DispatchTensor::Autodiff(_) => panic!("Operation not marked for autodiff.")
        }
    }};
}

/// Backend dispatch for unary operations.
///
/// When the return `=> Kind` is not provided, the operation output is not wrapped in a dispatch tensor (e.g., `into_data(..)`)
macro_rules! unary_op {
    ($tensor:expr, $inner_kind:ident, |$inner:ident| $body:expr => $kind:ident) => {
        backend_list!(unary_op_arms, $kind, $inner_kind, $tensor, |$inner| {
            $body
        })
    };
    ($tensor:expr, $inner_kind:ident, |$inner:ident| $body:expr) => {
        backend_list!(unary_op_arms, $inner_kind, $tensor, |$inner| { $body })
    };
}

/// Match arm generator for `unary_float`.
///
/// Similar to `unary_op_arms`, but float tensors are checked for autodiff support.
macro_rules! unary_float_arms {
    (
        $mode:ident, // `owned` or `ref`
        $kind:ident,
        $inner_kind:ident,
        $tensor:expr,
        |$inner:ident| $body:expr;
        $([$Backend:ident, $cfg:meta]),*
    ) => {{
        match $tensor {
            #[cfg(feature = "autodiff")]
            $crate::DispatchTensor::Autodiff(inner) => {
                unary_float_arms!(
                    @autodiff $mode,
                    $kind,
                    { if_mode!($mode, &**inner, *inner) },
                    |$inner| $body;
                    $([$Backend, $cfg]),*
                )
            },
            $(
                #[cfg($cfg)]
                $crate::DispatchTensor::$Backend($inner) => {
                    type B = $Backend<f32>;
                    let $inner = unary_float_arms!(@unwrap $mode, $inner, $inner_kind);
                    $crate::DispatchTensor::$Backend(
                        $crate::BackendTensor::$kind($body)
                    )
                }
            )*
        }
    }};

    // --- Autodiff recursive arm ---
    (
        @autodiff $mode:ident,
        $kind:ident,
        $tensor:expr,
        |$inner:ident| $body:expr;
        $([$Backend:ident, $cfg:meta]),*
    ) => {{
        match $tensor {
            $(
                #[cfg($cfg)]
                $crate::DispatchTensor::$Backend($inner) => {
                    type B = Autodiff<$Backend<f32>>;
                    let $inner = unary_float_arms!(@unwrap_ad $mode, $inner);
                    wrap_float!( @wrap_autodiff $kind, $Backend, { $body } )
                }
            )*
            $crate::DispatchTensor::Autodiff(_) => panic!("Autodiff should not wrap an autodiff tensor.")
        }
    }};

    // --- Non-wrapping arms (operations not returning a tensor) ---
    (
        $mode:ident,
        $inner_kind:ident,
        $tensor:expr,
        |$inner:ident| $body:expr;
        $([$Backend:ident, $cfg:meta]),*
    ) => {{
        match $tensor {
            #[cfg(feature = "autodiff")]
            $crate::DispatchTensor::Autodiff(inner) => {
                unary_float_arms!(
                    @autodiff $mode,
                    { if_mode!($mode, &**inner, *inner) },
                    |$inner| $body;
                    $([$Backend, $cfg]),*
                )
            },
            $(
                #[cfg($cfg)]
                $crate::DispatchTensor::$Backend($inner) => {
                    type B = $Backend<f32>;
                    let $inner = unary_float_arms!(@unwrap $mode, $inner, $inner_kind);
                    $body
                }
            )*
        }
    }};
    (
        @autodiff $mode:ident,
        $tensor:expr,
        |$inner:ident| $body:expr;
        $([$Backend:ident, $cfg:meta]),*
    ) => {{
        match $tensor {
            $(
                #[cfg($cfg)]
                $crate::DispatchTensor::$Backend($inner) => {
                    type B = Autodiff<$Backend<f32>>;
                    let $inner = unary_float_arms!(@unwrap_ad $mode, $inner);
                    $body
                }
            )*
            $crate::DispatchTensor::Autodiff(_) => panic!("Autodiff should not wrap an autodiff tensor.")
        }
    }};

    // --- Helpers to unwarp the tensor based on owned/ref ---
    (@unwrap owned, $inner:ident, $inner_kind:ident) => { $inner.$inner_kind() };
    (@unwrap ref, $inner:ident, $inner_kind:ident) => {
        paste::paste! { $inner.[< as_ $inner_kind >]() }
    };

    (@unwrap_ad owned, $inner:ident) => { $inner.autodiff() };
    (@unwrap_ad ref, $inner:ident) => { $inner.as_autodiff() };

}

/// Utility to pick a token based on mode
macro_rules! if_mode {
    (ref, $if_ref:expr, $if_owned:expr) => {
        $if_ref
    };
    (owned, $if_ref:expr, $if_owned:expr) => {
        $if_owned
    };
}

/// Backend dispatch for float unary operations (that might support autodiff).
///
/// When the return `=> Kind` is not provided, the operation output is not wrapped in a dispatch tensor (e.g., `into_data(..)`)
macro_rules! unary_float {
    // Owned with return kind
    ($tensor:expr, $inner_kind:ident, |$inner:ident| $body:expr => $kind:ident) => {
        backend_list!(
            unary_float_arms,
            owned,
            $kind,
            $inner_kind,
            $tensor,
            |$inner| { $body }
        )
    };
    // Owned without return kind
    ($tensor:expr, $inner_kind:ident, |$inner:ident| $body:expr) => {
        backend_list!(unary_float_arms, owned, $inner_kind, $tensor, |$inner| {
            $body
        })
    };
    // Reference without return kind
    (ref $tensor:expr, $inner_kind:ident, |$inner:ident| $body:expr) => {
        backend_list!(unary_float_arms, ref, $inner_kind, $tensor, |$inner| {
            $body
        })
    };
}
/// Match arm generator for `binary_op`.
/// Matches two tensors to ensure they share the same backend before unwrapping them for the operation.
macro_rules! binary_op_arms {
    (
        $kind:ident,
        ($lhs:expr, $lhs_kind:ident),
        ($rhs:expr, $rhs_kind:ident),
        |$lhs_inner:ident, $rhs_inner:ident| $body:expr;
        $([$Backend:ident, $cfg:meta]),*
    ) => {{
        match ($lhs, $rhs) {
            $(
                #[cfg($cfg)]
                ($crate::DispatchTensor::$Backend($lhs_inner), $crate::DispatchTensor::$Backend($rhs_inner)) => {
                    type B = $Backend<f32>;
                    let $lhs_inner = $lhs_inner.$lhs_kind();
                    let $rhs_inner = $rhs_inner.$rhs_kind();
                    $crate::DispatchTensor::$Backend($crate::BackendTensor::$kind($body))
                }
            )*
            (lhs, rhs) => {
                panic!(
                    "The provided tensors are not on the same backend. Got backends {:?} and {:?}.", lhs, rhs
                );
            }
        }
    }};
}

/// Backend dispatch for binary operations.
/// Automatically verifies that both tensors reside on the same backend.
macro_rules! binary_op {
    (($lhs:expr, $lhs_kind:ident), ($rhs:expr, $rhs_kind:ident), |$lhs_inner:ident, $rhs_inner:ident| $body:expr => $kind:ident) => {
        backend_list!(
            binary_op_arms,
            $kind,
            ($lhs, $lhs_kind),
            ($rhs, $rhs_kind),
            |$lhs_inner, $rhs_inner| { $body }
        )
    };
}

/// Match arm generator for `binary_float`.
/// Matches two tensors to ensure they share the same backend before unwrapping them for the operation.
macro_rules! binary_float_arms {
    // (float, float) binary op
    (
        $kind:ident,
        ($lhs:expr, float),
        ($rhs:expr, float),
        |$lhs_inner:ident, $rhs_inner:ident| $body:expr;
        $([$Backend:ident, $cfg:meta]),*
    ) => {{
        match ($lhs, $rhs) {
            // Autodiff arms first
            #[cfg(feature = "autodiff")]
            ($crate::DispatchTensor::Autodiff(lhs_inner), $crate::DispatchTensor::Autodiff(rhs_inner)) => {
                // Recursively dispatch on inner
                binary_float_arms!(
                    @autodiff
                    $kind,
                    (*lhs_inner, autodiff),
                    (*rhs_inner, autodiff),
                    |$lhs_inner, $rhs_inner| $body;
                    $([$Backend, $cfg]),*
                )
            },
            $(
                #[cfg($cfg)]
                ($crate::DispatchTensor::$Backend($lhs_inner), $crate::DispatchTensor::$Backend($rhs_inner)) => {
                    type B = $Backend<f32>;
                    let $lhs_inner = $lhs_inner.float();
                    let $rhs_inner = $rhs_inner.float();
                    $crate::DispatchTensor::$Backend($crate::BackendTensor::$kind($body))
                }
            )*
            (lhs, rhs) => {
                panic!(
                    "The provided tensors are not on the same backend. Got backends {:?} and {:?}.", lhs, rhs
                );
            }
        }
    }};
    // (float, any) binary op
    (
        $kind:ident,
        ($lhs:expr, float),
        ($rhs:expr, $rhs_kind:ident),
        |$lhs_inner:ident, $rhs_inner:ident| $body:expr;
        $([$Backend:ident, $cfg:meta]),*
    ) => {{
        match ($lhs, $rhs) {
            $(
                // Autodiff arms first
                #[cfg(all(feature = "autodiff", $cfg))]
                ($crate::DispatchTensor::Autodiff(lhs_inner), $crate::DispatchTensor::$Backend($rhs_inner)) => {
                    // Match on inner
                    match *lhs_inner {
                        $crate::DispatchTensor::$Backend($lhs_inner) => {
                            type B = Autodiff<$Backend<f32>>;
                            let $lhs_inner = $lhs_inner.autodiff();
                            let $rhs_inner = $rhs_inner.$rhs_kind();
                            wrap_float!(
                                @wrap_autodiff
                                $kind,
                                $Backend,
                                { $body }
                            )
                        }
                        $crate::DispatchTensor::Autodiff(_) => panic!("Autodiff should not wrap an autodiff tensor."),
                        _ => panic!("The provided tensors are not on the same backend.")
                    }
                },

                #[cfg($cfg)]
                ($crate::DispatchTensor::$Backend($lhs_inner), $crate::DispatchTensor::$Backend($rhs_inner)) => {
                    type B = $Backend<f32>;
                    let $lhs_inner = $lhs_inner.float();
                    let $rhs_inner = $rhs_inner.$rhs_kind();
                    $crate::DispatchTensor::$Backend($crate::BackendTensor::$kind($body))
                }
            )*
            (lhs, rhs) => {
                panic!(
                    "The provided tensors are not on the same backend. Got backends {:?} and {:?}.", lhs, rhs
                );
            }
        }
    }};
    (
        $kind:ident,
        ($lhs:expr, $lhs_kind:ident),
        ($rhs:expr, $rhs_kind:ident),
        |$lhs_inner:ident, $rhs_inner:ident| $body:expr;
        $([$Backend:ident, $cfg:meta]),*
    ) => {{
        match ($lhs, $rhs) {
            $(
                #[cfg($cfg)]
                ($crate::DispatchTensor::$Backend($lhs_inner), $crate::DispatchTensor::$Backend($rhs_inner)) => {
                    type B = $Backend<f32>;
                    let $lhs_inner = $lhs_inner.$lhs_kind();
                    let $rhs_inner = $rhs_inner.$rhs_kind();
                    $crate::DispatchTensor::$Backend($crate::BackendTensor::$kind($body))
                }
            )*
            (lhs, rhs) => {
                panic!(
                    "The provided tensors are not on the same backend. Got backends {:?} and {:?}.", lhs, rhs
                );
            }
        }
    }};
    // Autodiff (lhs, rhs) tensors
    (
        @autodiff
        $kind:ident,
        ($lhs:expr, $lhs_kind:ident),
        ($rhs:expr, $rhs_kind:ident),
        |$lhs_inner:ident, $rhs_inner:ident| $body:expr;
        $([$Backend:ident, $cfg:meta]),*
    ) => {{
        match ($lhs, $rhs) {
            $(
                #[cfg($cfg)]
                ($crate::DispatchTensor::$Backend($lhs_inner), $crate::DispatchTensor::$Backend($rhs_inner)) => {
                    type B = Autodiff<$Backend<f32>>;
                    let $lhs_inner = $lhs_inner.$lhs_kind();
                    let $rhs_inner = $rhs_inner.$rhs_kind();
                    wrap_float!(
                        @wrap_autodiff
                        $kind,
                        $Backend,
                        { $body }
                    )
                }
            )*
            #[cfg(feature = "autodiff")]
            ($crate::DispatchTensor::Autodiff(_), _) | (_, $crate::DispatchTensor::Autodiff(_))  => panic!("Autodiff should not wrap an autodiff tensor."),
            (lhs, rhs) => {
                panic!(
                    "The provided tensors are not on the same backend. Got backends {:?} and {:?}.", lhs, rhs
                );
            }
        }
    }};

}

/// Backend dispatch for binary operations.
/// Automatically verifies that both tensors reside on the same backend.
macro_rules! binary_float {
    (($lhs:expr, $lhs_kind:ident), ($rhs:expr, $rhs_kind:ident), |$lhs_inner:ident, $rhs_inner:ident| $body:expr => $kind:ident) => {
        backend_list!(
            binary_float_arms,
            $kind,
            ($lhs, $lhs_kind),
            ($rhs, $rhs_kind),
            |$lhs_inner, $rhs_inner| { $body }
        )
    };
}

/// The core logic for a single backend in a `multi_op`.
/// Handles the manual unwrapping of required/optional inputs and the
/// re-wrapping of multiple required/optional output tensors.
macro_rules! multi_op_arm {
    (
        $Backend:ident,
        [ $( ($x:ident, $x_kind:ident) ),+ ],
        [ $( ($opt_in:ident, $opt_kind:ident) ),* ],
        [ $( ($out:ident, $out_kind:ident) ),+  ],
        [ $( $opt_out:ident ),* ],
        $body:expr
    ) => {{
        type B = $Backend<f32>;

        // Required inputs
        $(
            let $x = match $x {
                $crate::DispatchTensor::$Backend(inner) => inner.$x_kind(),
                _ => panic!("Input tensor {} is on the wrong device", stringify!($x)),
            };
        )+

        // Optional inputs
        $(
            let $opt_in = $opt_in.map(|o| match o {
                $crate::DispatchTensor::$Backend(inner) => inner.$opt_kind(),
                _ => panic!("Optional tensor {} is on the wrong device", stringify!($opt_in)),
            });
        )*

        let ($($out),+, $($opt_out),*) = $body;

        // Outputs and optional outputs
        (
            $( $crate::DispatchTensor::$Backend($crate::BackendTensor::$out_kind($out)) ),+,
            $( $opt_out.map(|t| $crate::DispatchTensor::$Backend($crate::BackendTensor::Float(t))) ),*
        )
    }};
}

macro_rules! wrap_input_autodiff {
    ($Backend:ident, $inner:expr, int) => {
        $inner.int()
    };
    ($Backend:ident, $inner:expr, bool) => {
        $inner.bool()
    };
    // Float tensors: wrap with autodiff
    ($Backend:ident, $inner:expr, float) => {
        $inner.autodiff()
    };
}

// DispatchTensor::Autodiff(DispatchTensor::$Backend(BackendTensor::Autodiff()))
macro_rules! multi_op_arm_autodiff {
    (
        $Backend:ident,
        [ $( ($x:ident, $x_kind:ident) ),+ ],
        [ $( ($opt_in:ident, $opt_kind:ident) ),* ],
        [ $( ($out:ident, $out_kind:ident) ),+  ],
        [ $( $opt_out:ident ),* ],
        $body:expr
    ) => {{
        type B = Autodiff<$Backend<f32>>;

        // Required inputs
        $(
            let $x = match $x {
                $crate::DispatchTensor::Autodiff(inner) => {
                    match *inner {
                        $crate::DispatchTensor::$Backend(inner) => wrap_input_autodiff!($Backend, inner, $x_kind),
                        _ => panic!("Input tensor {} is on the wrong device", stringify!($x)),
                    }
                },
                // Unreachable, except when input is int
                $crate::DispatchTensor::$Backend(inner) => wrap_input_autodiff!($Backend, inner, $x_kind),
                _ => panic!("Input tensor {} is on the wrong device", stringify!($x)),
            };
        )+

        // Optional inputs (always assumed to be float / autodiff)
        $(
            let $opt_in = $opt_in.map(|o| match o {
                $crate::DispatchTensor::Autodiff(inner) => {
                    match *inner {
                        $crate::DispatchTensor::$Backend(inner) => wrap_input_autodiff!($Backend, inner, $opt_kind),
                        _ => panic!("Input tensor {} is on the wrong device", stringify!($opt_in)),
                    }
                },
                _ => panic!("Optional tensor {} is on the wrong device", stringify!($opt_in)),
            });
        )*

        let ($($out),+, $($opt_out),*) = $body;

        // Outputs and optional outputs
        (
            $( wrap_float!(@wrap_autodiff $out_kind, $Backend, $out) ),+,
            $( $opt_out.map(|t| wrap_float!(@wrap_autodiff Float, $Backend, t)) ),*
        )
    }};
}

/// Helper to extract the first identifier from an input list.
/// Used to determine the device/backend for dispatching multi-tensor operations.
macro_rules! first_input {
    ([ ($x:ident, $kind:ident) $(, $rest:tt)* ]) => {
        $x
    };
}

/// Match arm generator for `multi_op`.
/// Determines the backend based on the first input and delegates to `multi_op_arm`
/// to handle the repetition-heavy unwrapping and wrapping logic.
macro_rules! multi_op_arms_autodiff {
    (
        $inputs:tt,
        $opt_inputs:tt,
        $outputs:tt,
        $opt_outputs:tt,
        $body:expr;
        $( [$Backend:ident, $cfg:meta] ),*
    ) => {{
        match &first_input!($inputs) {
            // Autodiff first
            #[cfg(feature = "autodiff")]
            $crate::DispatchTensor::Autodiff(inner) => {
                match **inner {
                    $(
                        #[cfg($cfg)]
                        $crate::DispatchTensor::$Backend(_) => {
                            multi_op_arm_autodiff!(
                                $Backend,
                                $inputs,
                                $opt_inputs,
                                $outputs,
                                $opt_outputs,
                                $body
                            )
                        }
                    )*
                    $crate::DispatchTensor::Autodiff(_) => panic!("Autodiff should not wrap an autodiff tensor.")
                }
            },
            $(
                #[cfg($cfg)]
                $crate::DispatchTensor::$Backend(_) => {
                    multi_op_arm!(
                        $Backend,
                        $inputs,
                        $opt_inputs,
                        $outputs,
                        $opt_outputs,
                        $body
                    )
                }
            )*
        }
    }};
}

/// Match arm generator for `multi_op`.
///
/// Similar to `multi_op_arms`, but skips autodiff checks.
macro_rules! multi_op_arms {
    (
        $inputs:tt,
        $opt_inputs:tt,
        $outputs:tt,
        $opt_outputs:tt,
        $body:expr;
        $( [$Backend:ident, $cfg:meta] ),*
    ) => {{
        match &first_input!($inputs) {
            $(
                #[cfg($cfg)]
                $crate::DispatchTensor::$Backend(_) => {
                    multi_op_arm!(
                        $Backend,
                        $inputs,
                        $opt_inputs,
                        $outputs,
                        $opt_outputs,
                        $body
                    )
                }
            )*
            #[cfg(feature = "autodiff")]
            $crate::DispatchTensor::Autodiff(_) => panic!("Operation not marked for autodiff.")
        }
    }};
}

/// High-level macro for complex module operations (e.g., conv2d) and multi-tensor operations.
/// Handles variable numbers of required/optional inputs and wraps multiple outputs.
///
/// Usage:
/// ```ignore
/// multi_op!(
///     inputs[(x, float), (weight, float)],
///     opt_inputs[(bias, float)],
///     => Float,
///     B::conv2d(x, weight, bias, options)
/// )
/// ```
macro_rules! multi_op {
    // --- Single output shorthands ---
    // Automatically wraps body in tuple and extracts .0
    (
        inputs[$( ($x:ident, $kind:ident) ),+],
        => Float,
        $body:expr
    ) => {
        multi_op!(
            inputs[$( ($x, $kind) ),+],
            opt_inputs[],
            outputs[(out, Float)],
            opt_outputs[],
            { ($body,) }
        )
        .0
    };
    (
        inputs[$( ($x:ident, $kind:ident) ),+],
        opt_inputs[ $(($opt_in:ident, $opt_kind:ident)),* ],
        => $out_kind:ident,
        $body:expr
    ) => {
        multi_op!(
            inputs[$( ($x, $kind) ),+],
            opt_inputs[ $(($opt_in, $opt_kind)),* ],
            outputs[(out, $out_kind)],
            opt_outputs[],
            { ($body,) }
        )
        .0
    };
    // Int/Bool op specialization (not marked for autodiff)
    (
        inputs[$( ($x:ident, $kind:ident) ),+],
        => $out_kind:ident,
        $body:expr
    ) => {
        backend_list!(
            multi_op_arms,
            [ $(($x, $kind)),+ ],
            [],
            [ (out, $out_kind) ],
            [],
            { ($body,) }
        ).0
    };

    // --- Required + optional for both inputs and outputs ---
    (
        inputs[ $(($x:ident, $kind:ident)),+ ],
        opt_inputs[ $(($opt_in:ident, $opt_kind:ident)),* ],
        outputs[ $( ($out:ident, $out_kind:ident) ),+ ],
        opt_outputs[ $($opt_out:ident),* ],
        $body:expr
    ) => {
        backend_list!(
            multi_op_arms_autodiff,
            [ $(($x, $kind)),+ ],
            [ $(($opt_in, $opt_kind)),* ],
            [ $(($out, $out_kind)),+ ],
            [ $($opt_out),* ],
            $body
        )
    };

    (
        inputs[ $(($x:ident, $kind:ident)),+ ],
        opt_inputs[ $(($opt_in:ident, $opt_kind:ident)),* ],
        outputs[ $($out:ident),+ ],
        $body:expr
    ) => {
        multi_op!(
            inputs[ $(($x, $kind)),+ ],
            opt_inputs[ $(($opt_in, $opt_kind)),* ],
            outputs[ $(($out, Float)),+ ],
            opt_outputs[],
            $body
        )
    };

    (
        inputs[ $(($x:ident, $kind:ident)),+ ],
        outputs[ $( ($out:ident, $out_kind:ident) ),+ ],
        $body:expr
    ) => {
        multi_op!(
            inputs[ $(($x, $kind)),+ ],
            opt_inputs[],
            outputs[ $(($out, $out_kind)),+ ],
            opt_outputs[],
            $body
        )
    };
}

/// Unwraps a `Vec<DispatchTensor>` for a known backend.
macro_rules! unwrap_vec {
    ($Backend:ident, $vec:expr, $kind:ident) => {
        $vec.into_iter()
            .map(|t| match t {
                $crate::DispatchTensor::$Backend(inner) => inner.$kind(),
                _ => panic!(
                    "Tensor is on the wrong backend (expected {}).",
                    stringify!($Backend)
                ),
            })
            .collect::<Vec<_>>()
    };

    // Autodiff-wrapped backend
    (@autodiff $Backend:ident, $vec:expr, $kind:ident) => {
        $vec.into_iter()
            .map(|t| match t {
                $crate::DispatchTensor::Autodiff(inner) => match *inner {
                    $crate::DispatchTensor::$Backend(inner) => inner.$kind(),
                    _ => panic!(
                        "Autodiff float tensor is on the wrong backend (expected {}).",
                        stringify!($Backend)
                    ),
                },
                _ => panic!(
                    "Expected autodiff-wrapped float tensor for backend {}.",
                    stringify!($Backend)
                ),
            })
            .collect::<Vec<_>>()
    };
}

/// Match arm generator for `vec_op`.
macro_rules! vec_op_arms {
    (Float, $inner_kind:ident, $tensors:expr, |$inner:ident| $body:expr; $([$Backend:ident, $cfg:meta]),*) => {
        match &$tensors[0] {
            // Autodiff arm first
            #[cfg(feature = "autodiff")]
            $crate::DispatchTensor::Autodiff(inner) => {
                // Recursively dispatch on inner
                match **inner {
                    $(
                    #[cfg($cfg)]
                    $crate::DispatchTensor::$Backend(_) => {
                        type B = Autodiff<$Backend<f32>>;

                        let $inner = unwrap_vec!(@autodiff $Backend, $tensors, autodiff);
                        wrap_float!( @wrap_autodiff Float, $Backend, { $body } )
                    }
                )*
                    $crate::DispatchTensor::Autodiff(_) => panic!("Autodiff should not wrap an autodiff tensor.")
                }
            },

            $(
                #[cfg($cfg)]
                $crate::DispatchTensor::$Backend(_) => {
                    type B = $Backend<f32>;

                    let $inner = unwrap_vec!($Backend, $tensors, $inner_kind);
                    $crate::DispatchTensor::$Backend($crate::BackendTensor::Float($body))
                }
            )*
        }
    };
    ($kind:ident, $inner_kind:ident, $tensors:expr, |$inner:ident| $body:expr; $([$Backend:ident, $cfg:meta]),*) => {
        match &$tensors[0] {
            $(
                #[cfg($cfg)]
                $crate::DispatchTensor::$Backend(_) => {
                    type B = $Backend<f32>;

                    let $inner = unwrap_vec!($Backend, $tensors, $inner_kind);
                    $crate::DispatchTensor::$Backend($crate::BackendTensor::$kind($body))
                }
            )*
            #[cfg(feature = "autodiff")]
            $crate::DispatchTensor::Autodiff(_) => panic!("Operation not marked for autodiff.")
        }
    };
}

/// Backend dispatch for operations on multiple inputs (vec).
/// Automatically verifies that tensors reside on the first backend.
macro_rules! vec_op {
    ($tensors:expr, $inner_kind:ident, |$inner:ident| $body:expr => $kind:ident) => {
        backend_list!(vec_op_arms, $kind, $inner_kind, $tensors, |$inner| {
            $body
        })
    };
}

/// Match arm generator for `transaction_op`.
macro_rules! transaction_op_arms {
    ($tx:ident, $first:expr; $([$Backend:ident, $cfg:meta]),*) => {
        match $first {
            // Autodiff arm first
            #[cfg(feature = "autodiff")]
            $crate::DispatchTensor::Autodiff(inner) => {
                // Recursively dispatch on inner
                match **inner {
                    $(
                    #[cfg($cfg)]
                    $crate::DispatchTensor::$Backend(_) => {
                        type B = $Backend<f32>;

                        // Unwrap vec
                        let floats = unwrap_vec!(@autodiff $Backend, $tx.read_floats, autodiff_inner);
                        let ints = unwrap_vec!($Backend, $tx.read_ints, int);
                        let bools = unwrap_vec!($Backend, $tx.read_bools, bool);
                        // Not supported
                        let qfloats = $tx.read_qfloats.into_iter().map(|_t| todo!("Quantization not supported yet")).collect();

                        B::tr_execute(TransactionPrimitive::new(floats, qfloats, ints, bools)).await
                    }
                )*
                    $crate::DispatchTensor::Autodiff(_) => panic!("Autodiff should not wrap an autodiff tensor.")
                }
            },

            $(
                #[cfg($cfg)]
                $crate::DispatchTensor::$Backend(_) => {
                    type B = $Backend<f32>;

                    // Unwrap vec
                    let floats = unwrap_vec!($Backend, $tx.read_floats, float);
                    let ints = unwrap_vec!($Backend, $tx.read_ints, int);
                    let bools = unwrap_vec!($Backend, $tx.read_bools, bool);
                    // Not supported
                    let qfloats = $tx.read_qfloats.into_iter().map(|_t| todo!("Quantization not supported yet")).collect();

                    B::tr_execute(TransactionPrimitive::new(floats, qfloats, ints, bools)).await
                }
            )*
        }
    };
}

/// Helper to dispatch a transaction based on the first available tensor.
macro_rules! transaction_op {
    ($tx:ident, $first:expr) => {
        backend_list!(transaction_op_arms, $tx, $first)
    };
}
