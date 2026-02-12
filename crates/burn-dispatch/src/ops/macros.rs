// TODO:
// 1. Document macro usage
// 2. Simplify other dispatch macros based on `apply_to_device` with auto generated combinations (same backend only)

/// Supplies a list of all supported backends and their corresponding feature flags
/// to a callback macro. This centralizes the backend registry.
#[macro_export]
macro_rules! backend_list {
    ($callback:ident, $($extra:tt)*) => {
        $crate::$callback! {
            $($extra)*;
            [Cpu, "cpu"],
            [Cuda, "cuda"],
            [Metal, "metal"],
            [Rocm, "rocm"],
            [Vulkan, "vulkan"],
            [WebGpu, "webgpu"],
            [NdArray, "ndarray"],
            [LibTorch, "tch"]
        }
    };
}

/// Supplies a matrix of cross-backend combinations. Used for operations where the source and destination backends may differ.
#[macro_export]
macro_rules! backend_matrix {
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
macro_rules! dispatch_device_arms {
    (
        $device:expr,
        |$inner:ident| $body:expr;
        $([$Backend:ident, $feature:literal]),*
    ) => {
        match $device {
            $(
                #[cfg(feature = $feature)]
                $crate::Device::$Backend($inner) => {
                    type B = $Backend<f32>;
                    $body
                }
            )*
        }
    };
}

/// Dispatches an operation body based on the provided device.
#[macro_export]
macro_rules! dispatch_device {
    ($device:expr, |$inner:ident| $body:expr) => {
        $crate::backend_list!(dispatch_device_arms, $device, |$inner| $body)
    };
}

#[macro_export]
macro_rules! to_device_arms {
    (
        $kind:ident, $inner_fn:ident, $tensor:expr, $device:expr, $to_device:ident, |$inner:ident, $device_ident:ident| $body:expr;
        $( [$B1:ident, $src_feature:literal] => [ $( [$B2:ident, $dst_feature:literal] ),+ ] );*
    ) => {
        match ($tensor, $device) {
            // --- Same backend to_device ---
            $(
                #[cfg(feature = $src_feature)]
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
                    #[cfg(all(feature = $src_feature, feature = $dst_feature))]
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

            _ => todo!("Requested device combination is not supported or feature-gated off"),
        }
    };
}

/// Handles tensor movement between devices, supporting both same-backend transfers
/// and cross-backend dispatches.
#[macro_export]
macro_rules! to_device {
    ($kind:ident, $inner_fn:ident, $tensor:expr, $device:expr, $to_device:ident, |$inner:ident, $device_ident:ident| $body:expr) => {
        $crate::backend_matrix!(
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

#[macro_export]
macro_rules! creation_op_arms {
    (
        $kind:ident,
        $device:expr,
        |$inner:ident| $body:expr;
        $([$Backend:ident, $feature:literal]),*
    ) => {{
        match $device {
            $(
                #[cfg(feature = $feature)]
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
#[macro_export]
macro_rules! creation_op {
    ($kind:ident, $device:expr, |$inner:ident| $body:expr) => {
        $crate::backend_list!(creation_op_arms, $kind, $device, |$inner| $body)
    };
}

#[macro_export]
macro_rules! unary_op_arms {
    (
        $kind:ident,
        $inner_kind:ident,
        $tensor:expr,
        |$inner:ident| $body:expr;
        $([$Backend:ident, $feature:literal]),*
    ) => {{
        match $tensor {
            $(
                #[cfg(feature = $feature)]
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
        $([$Backend:ident, $feature:literal]),*
    ) => {{
        match $tensor {
            $(
                #[cfg(feature = $feature)]
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
#[macro_export]
macro_rules! unary_op {
    ($tensor:expr, $inner_kind:ident, |$inner:ident| $body:expr => $kind:ident) => {
        $crate::backend_list!(unary_op_arms, $kind, $inner_kind, $tensor, |$inner| {
            $body
        })
    };
    ($tensor:expr, $inner_kind:ident, |$inner:ident| $body:expr) => {
        $crate::backend_list!(unary_op_arms, $inner_kind, $tensor, |$inner| { $body })
    };
}

#[macro_export]
macro_rules! binary_op_arms {
    (
        $kind:ident,
        ($lhs:expr, $lhs_kind:ident),
        ($rhs:expr, $rhs_kind:ident),
        |$lhs_inner:ident, $rhs_inner:ident| $body:expr;
        $([$Backend:ident, $feature:literal]),*
    ) => {{
        match ($lhs, $rhs) {
            $(
                #[cfg(feature = $feature)]
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
#[macro_export]
macro_rules! binary_op {
    (($lhs:expr, $lhs_kind:ident), ($rhs:expr, $rhs_kind:ident), |$lhs_inner:ident, $rhs_inner:ident| $body:expr => $kind:ident) => {
        $crate::backend_list!(
            binary_op_arms,
            $kind,
            ($lhs, $lhs_kind),
            ($rhs, $rhs_kind),
            |$lhs_inner, $rhs_inner| { $body }
        )
    };
}

#[macro_export]
macro_rules! multi_op_arms {
    (
        $kind:ident,
        ($t1:expr, $t1_kind:ident),
        ($t2:expr, $t2_kind:ident),
        ($t3:expr, $t3_kind:ident),
        |$t1_inner:ident, $t2_inner:ident, $t3_inner:ident| $body:expr;
        $([$Backend:ident, $feature:literal]),*
    ) => {{
        match ($t1, $t2, $t3) {
            $(
                #[cfg(feature = $feature)]
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
#[macro_export]
macro_rules! multi_tensor_op {
    (($t1:expr, $t1_kind:ident), ($t2:expr, $t2_kind:ident), ($t3:expr, $t3_kind:ident), |$t1_inner:ident, $t2_inner:ident, $t3_inner:ident| $body:expr => $kind:ident) => {
        $crate::backend_list!(
            multi_op_arms,
            $kind,
            ($t1, $t1_kind),
            ($t2, $t2_kind),
            ($t3, $t3_kind),
            |$t1_inner, $t2_inner, $t3_inner| { $body }
        )
    };
}

#[macro_export]
macro_rules! module_op_arm {
    (
        $Backend:ident,
        [ $( ($x:ident, $kind:ident) ),+ ],
        [ $( $opt_in:ident ),* ],
        [ $( $out:ident ),+ ],
        [ $( $opt_out:ident ),* ],
        $body:expr
    ) => {{
        type B = $Backend<f32>;

        // Required inputs
        $(
            let $x = match $x {
                $crate::DispatchTensor::$Backend(inner) => inner.$kind(),
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
            $( $crate::DispatchTensor::$Backend($crate::as_float($out)) ),+,
            $( $opt_out.map(|t| $crate::DispatchTensor::$Backend($crate::as_float(t))) ),*
        )
    }};
}

// Helper to extract the first identifier for the match statement
#[macro_export]
macro_rules! first_input {
    ([ ($x:ident, $kind:ident) $(, $rest:tt)* ]) => {
        $x
    };
}

#[macro_export]
macro_rules! module_op_arms {
    (
        $inputs:tt,
        $opt_inputs:tt,
        $outputs:tt,
        $opt_outputs:tt,
        $body:expr;
        $( [$Backend:ident, $feature:literal] ),*
    ) => {
        // #[allow(unused_parens, unreachable_patterns)]
        match $crate::first_input!($inputs) {
            $(
                #[cfg(feature = $feature)]
                $crate::DispatchTensor::$Backend(_) => {
                    $crate::module_op_arm!(
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
/// ```rust
/// module_op!(
///     inputs[(x, float), (weight, float)],
///     opt_inputs[bias] =>
///     outputs[out]
///     { (B::conv2d(x, weight, bias, options),) }
/// )
/// ```
// NOTE: a bit verbose but easier to handle automatic backend and input repetitions.
#[macro_export]
macro_rules! module_op {
    // Required inputs + required outputs
    (
        inputs[ $(($x:ident, $kind:ident)),+ ],
        opt_inputs[ $($opt_in:ident),* ] =>
        outputs[ $($out:ident),+ ]
        $body:expr
    ) => {
        $crate::module_op!(
            inputs[ $(($x, $kind)),+ ],
            opt_inputs[ $($opt_in),* ] =>
            outputs[ $($out),+ ],
            opt_outputs[] // empty opt outputs
            $body
        )
    };

    // Required + optional for both inputs and outputs
    (
        inputs[ $(($x:ident, $kind:ident)),+ ],
        opt_inputs[ $($opt_in:ident),* ] =>
        outputs[ $($out:ident),+ ],
        opt_outputs[ $($opt_out:ident),* ]
        $body:expr
    ) => {
        $crate::backend_list!(
            module_op_arms,
            [ $(($x, $kind)),+ ],
            [ $($opt_in),* ],
            [ $($out),+ ],
            [ $($opt_out),* ],
            $body
        )
    };
}
