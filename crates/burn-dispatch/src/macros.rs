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
            $(
                #[cfg($cfg)]
                $crate::Device::$Backend($inner) => {
                    type B = $Backend<f32>;
                    $body
                }
            )*
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

/// Match arm generator for `creation_op`.
/// Matches a device and provides the specific backend type `B` to the closure before wrapping
/// the resulting tensor in the corresponding `DispatchTensor` variant.
macro_rules! creation_op_arms {
    (
        $kind:ident,
        $device:expr,
        |$inner:ident| $body:expr;
        $([$Backend:ident, $cfg:meta]),*
    ) => {{
        match $device {
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
}

/// Dispatches a tensor creation operation (e.g., zeros, ones) to the correct backend
/// based on the provided device.
macro_rules! creation_op {
    ($kind:ident, $device:expr, |$inner:ident| $body:expr) => {
        backend_list!(creation_op_arms, $kind, $device, |$inner| $body)
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

/// Match arm generator for `multi_tensor_op`.
/// Matches three tensors to ensure they share the same backend before unwrapping them for the operation.
macro_rules! multi_op_arms {
    (
        $kind:ident,
        ($t1:expr, $t1_kind:ident),
        ($t2:expr, $t2_kind:ident),
        ($t3:expr, $t3_kind:ident),
        |$t1_inner:ident, $t2_inner:ident, $t3_inner:ident| $body:expr;
        $([$Backend:ident, $cfg:meta]),*
    ) => {{
        match ($t1, $t2, $t3) {
            $(
                #[cfg($cfg)]
                ($crate::DispatchTensor::$Backend($t1_inner), $crate::DispatchTensor::$Backend($t2_inner), $crate::DispatchTensor::$Backend($t3_inner)) => {
                    type B = $Backend<f32>;
                    let $t1_inner = $t1_inner.$t1_kind();
                    let $t2_inner = $t2_inner.$t2_kind();
                    let $t3_inner = $t3_inner.$t3_kind();
                    $crate::DispatchTensor::$Backend($crate::BackendTensor::$kind($body))
                }
            )*
            (t1, t2, t3) => {
                panic!(
                    "The provided tensors are not on the same backend. Got backends {:?}, {:?} and {:?}.", t1, t2, t3
                );
            }
        }
    }};
}

/// Backend dispatch for operations involving three tensors.
/// Automatically verifies that all tensors reside on the same backend.
macro_rules! multi_tensor_op {
    (($t1:expr, $t1_kind:ident), ($t2:expr, $t2_kind:ident), ($t3:expr, $t3_kind:ident), |$t1_inner:ident, $t2_inner:ident, $t3_inner:ident| $body:expr => $kind:ident) => {
        backend_list!(
            multi_op_arms,
            $kind,
            ($t1, $t1_kind),
            ($t2, $t2_kind),
            ($t3, $t3_kind),
            |$t1_inner, $t2_inner, $t3_inner| { $body }
        )
    };
}

/// The core logic for a single backend in a `module_op`.
/// Handles the manual unwrapping of required/optional inputs and the
/// re-wrapping of multiple required/optional output tensors.
macro_rules! module_op_arm {
    (
        $Backend:ident,
        [ $( ($x:ident, $x_kind:ident) ),+ ],
        [ $( $opt_in:ident ),* ],
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
                $crate::DispatchTensor::$Backend(inner) => inner.float(), // standard for module ops
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

/// Helper to extract the first identifier from an input list.
/// Used to determine the device/backend for dispatching multi-tensor operations.
macro_rules! first_input {
    ([ ($x:ident, $kind:ident) $(, $rest:tt)* ]) => {
        $x
    };
}

/// Match arm generator for `module_op`.
/// Determines the backend based on the first input and delegates to `module_op_arm`
/// to handle the repetition-heavy unwrapping and wrapping logic.
macro_rules! module_op_arms {
    (
        $inputs:tt,
        $opt_inputs:tt,
        $outputs:tt,
        $opt_outputs:tt,
        $body:expr;
        $( [$Backend:ident, $cfg:meta] ),*
    ) => {
        // #[allow(unused_parens, unreachable_patterns)]
        match first_input!($inputs) {
            $(
                #[cfg($cfg)]
                $crate::DispatchTensor::$Backend(_) => {
                    module_op_arm!(
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
    };
}

/// High-level macro for complex module operations (e.g., Conv, Linear).
/// Handles variable numbers of required/optional inputs and wraps multiple outputs.
///
/// Usage:
/// ```ignore
/// module_op!(
///     inputs[(x, float), (weight, float)],
///     opt_inputs[bias] =>
///     outputs[out]
///     { (B::conv2d(x, weight, bias, options),) }
/// )
/// ```
// NOTE: a bit verbose but easier to handle automatic backend and input repetitions.
macro_rules! module_op {
    // Required + optional for both inputs and outputs
    (
        inputs[ $(($x:ident, $kind:ident)),+ ],
        opt_inputs[ $($opt_in:ident),* ],
        outputs[ $( ($out:ident, $out_kind:ident) ),+ ],
        opt_outputs[ $($opt_out:ident),* ],
        $body:expr
    ) => {
        backend_list!(
            module_op_arms,
            [ $(($x, $kind)),+ ],
            [ $($opt_in),* ],
            [ $(($out, $out_kind)),+ ],
            [ $($opt_out),* ],
            $body
        )
    };

    (
        inputs[ $(($x:ident, $kind:ident)),+ ],
        opt_inputs[ $($opt_in:ident),* ],
        outputs[ $($out:ident),+ ],
        $body:expr
    ) => {
        module_op!(
            inputs[ $(($x, $kind)),+ ],
            opt_inputs[ $($opt_in),* ],
            outputs[ $(($out, Float)),+ ],
            opt_outputs[],
            $body  // Don't wrap - pass through as-is
        )
    };

    (
        inputs[ $(($x:ident, $kind:ident)),+ ],
        outputs[ $( ($out:ident, $out_kind:ident) ),+ ],
        $body:expr
    ) => {
        module_op!(
            inputs[ $(($x, $kind)),+ ],
            opt_inputs[],
            outputs[ $(($out, $out_kind)),+ ],
            opt_outputs[],
            $body  // Don't wrap - pass through as-is
        )
    };

    (
        inputs[ $(($x:ident, $kind:ident)),+ ],
        outputs[ $($out:ident),+ ],
        $body:expr
    ) => {
        module_op!(
            inputs[ $(($x, $kind)),+ ],
            opt_inputs[],
            outputs[ $(($out, Float)),+ ],
            opt_outputs[],
            $body  // Don't wrap - pass through as-is
        )
    };
}
