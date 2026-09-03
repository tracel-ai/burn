// `burn-backend-extension` owns the authoritative backend catalog and derives the distributed list
// and transfer matrix from it. Keep only these consumer wrappers local and visible to
// rust-analyzer; having the proc macro generate `macro_rules!` definitions made IDE resolution
// unreliable.
macro_rules! backend_list {
    ($callback:ident, $($extra:tt)*) => {
        burn_backend_extension::backend_catalog!(backends, $callback; $($extra)*)
    };
}

macro_rules! distributed_backend_list {
    ($callback:ident, $($extra:tt)*) => {
        burn_backend_extension::backend_catalog!(distributed, $callback; $($extra)*)
    };
}

macro_rules! backend_matrix {
    ($callback:ident, $($extra:tt)*) => {
        burn_backend_extension::backend_catalog!(matrix, $callback; $($extra)*)
    };
}

#[cfg(feature = "autodiff")]
/// Helper to map the runtime strategy to the compile-time Autodiff generic.
macro_rules! with_autodiff_backend {
    ($Backend:ident, $checkpointing:expr, |$B:ident| $body:expr) => {
        match $checkpointing {
            $crate::GradientCheckpointingStrategy::Balanced => {
                type $B = $crate::backends::Autodiff<
                    $crate::backends::$Backend,
                    burn_autodiff::checkpoint::strategy::BalancedCheckpointing,
                >;
                $body
            }
            $crate::GradientCheckpointingStrategy::Disabled => {
                type $B = $crate::backends::Autodiff<
                    $crate::backends::$Backend,
                    burn_autodiff::checkpoint::strategy::NoCheckpointing,
                >;
                $body
            }
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
            $crate::DispatchDevice::Autodiff(inner) => {
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
                $crate::DispatchDevice::$Backend($inner) => {
                    type B = $crate::backends::$Backend;
                    $body
                }
            )*
            #[allow(unreachable_patterns)]
            other => panic!("Distributed operations are not supported for device {other:?}"),
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
                $crate::DispatchDevice::$Backend($inner) => {
                    type B = $crate::backends::Autodiff<$crate::backends::$Backend>;
                    $body
                }
            )*
            $crate::DispatchDevice::Autodiff(_) => unreachable!("Autodiff should not wrap an autodiff device."),
            #[allow(unreachable_patterns)]
            other => panic!("Distributed operations are not supported for device {other:?}"),
        }
    };
}

/// Dispatches an operation body based on the provided device.
macro_rules! dispatch_device {
    ($device:expr, |$inner:ident| $body:expr) => {
        dispatch_device!(@internal backend_list, $device, |$inner| $body)
    };
    (@distributed $device:expr, |$inner:ident| $body:expr) => {
        dispatch_device!(@internal distributed_backend_list, $device, |$inner| $body)
    };
    (@internal $list_macro:ident, $device:expr, |$inner:ident| $body:expr) => {
        $list_macro!(dispatch_device_arms, $device, |$inner| $body)
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
        #[allow(unreachable_patterns)]
        match ($tensor.kind, $device) {
            // Capture is deliberately absent from the cross-backend matrix: it accepts concrete
            // initializer values, but a captured tensor cannot be materialized on another backend.
            #[cfg(feature = "capture")]
            ($crate::DispatchTensorKind::Capture(t), $crate::DispatchDevice::Capture(d)) => {
                $crate::DispatchTensor {
                    kind: $crate::DispatchTensorKind::Capture($crate::BackendTensor::$kind(
                        $crate::backends::Capture::$to_device(t.$inner_fn(), d)
                    )),
                    autodiff: $tensor.autodiff,
                }
            }
            // --- Same backend to_device ---
            $(
                #[cfg($src_cfg)]
                ($crate::DispatchTensorKind::$B1(t), $crate::DispatchDevice::$B1(d)) => {
                    $crate::DispatchTensor {
                        kind: $crate::DispatchTensorKind::$B1($crate::BackendTensor::$kind(
                            $crate::backends::$B1::$to_device(t.$inner_fn(), d)
                        )),
                        autodiff: $tensor.autodiff,
                    }
                }
            )*

            // Any concrete backend can be materialized on the capture backend.
            // This is how ordinary module parameters and sample inputs become
            // locally retained capture initializers.
            $(
                #[cfg(all($src_cfg, feature = "capture"))]
                ($crate::DispatchTensorKind::$B1(t), $crate::DispatchDevice::Capture($device_ident)) => {
                    type B1 = $crate::backends::$B1;
                    type B2 = $crate::backends::Capture;
                    let $inner = t.$inner_fn();
                    $crate::DispatchTensor {
                        kind: $crate::DispatchTensorKind::Capture(
                            $crate::BackendTensor::$kind($body)
                        ),
                        autodiff: $tensor.autodiff,
                    }
                }
            )*

            // --- Cross backend arms ---
            // This loop generates the grid of combinations
            $(
                $(
                    #[cfg(all($src_cfg, $dst_cfg))]
                    ($crate::DispatchTensorKind::$B1(t), $crate::DispatchDevice::$B2($device_ident)) => {
                        type B1 = $crate::backends::$B1;
                        type B2 = $crate::backends::$B2;
                        let $inner = t.$inner_fn();

                        $crate::DispatchTensor {
                            kind: $crate::DispatchTensorKind::$B2(
                                $crate::BackendTensor::$kind($body)
                            ),
                            autodiff: $tensor.autodiff,
                        }
                    }
                )+
            )*
            // --- To autodiff ---
            // This can happen when moving a bool or int tensor to the device of a float autodiff tensor.
            // We move it to the inner device and preserve the checkpointing strategy.

            // --- Same backend to_device ---
            $(
                #[cfg(all($src_cfg, feature = "autodiff"))]
                ($crate::DispatchTensorKind::$B1(t), $crate::DispatchDevice::Autodiff(device_ad))
                if matches!(&*device_ad.inner, $crate::DispatchDevice::$B1(_)) => {
                    let $crate::DispatchDevice::$B1(d) = &*device_ad.inner else { unreachable!() };

                    $crate::DispatchTensor {
                        kind: $crate::DispatchTensorKind::$B1($crate::BackendTensor::$kind(
                            <$crate::backends::$B1>::$to_device(t.$inner_fn(), d)
                        )),
                        autodiff: $crate::DispatchAutodiffContext::Enabled(
                            device_ad.checkpointing,
                        ),
                    }
                }
            )*

            // --- Same backend to_device ---
            $(
                $(
                    #[cfg(all($src_cfg, $dst_cfg, feature = "autodiff"))]
                    ($crate::DispatchTensorKind::$B1(tensor), $crate::DispatchDevice::Autodiff(device_ad))
                    if matches!(&*device_ad.inner, $crate::DispatchDevice::$B2(_)) => {
                        let $crate::DispatchDevice::$B2($device_ident) = &*device_ad.inner else { unreachable!() };
                        type B1 = $crate::backends::$B1;
                        type B2 = $crate::backends::$B2;
                        let $inner = tensor.$inner_fn();

                        $crate::DispatchTensor {
                            kind: $crate::DispatchTensorKind::$B2(
                                $crate::BackendTensor::$kind($body)
                            ),
                            autodiff: $crate::DispatchAutodiffContext::Enabled(
                                device_ad.checkpointing,
                            ),
                        }

                    },
                )+
            )*
            #[cfg(feature = "autodiff")]
            (_, $crate::DispatchDevice::Autodiff(_)) => unreachable!("Autodiff should not wrap an autodiff device."),
            #[cfg(feature = "autodiff")]
            ($crate::DispatchTensorKind::Autodiff(..), _) => panic!("Operation not marked for autodiff."),
            // Capture is intentionally one-way: initialized values can be moved onto a
            // capture device, but captured tensors have no materialized data to move back.
            #[cfg(feature = "capture")]
            ($crate::DispatchTensorKind::Capture(_), _) => {
                panic!("Cannot move a tensor from a capture device")
            }
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
        #[allow(unreachable_patterns)]
        match ($tensor.kind, $device) {
            #[cfg(feature = "autodiff")]
            ($crate::DispatchTensorKind::Autodiff(kind), $crate::DispatchDevice::Autodiff(device)) => {
                let $crate::DispatchAutodiffContext::Enabled(ckp) = $tensor.autodiff else {
                    panic!("an autodiff float primitive must have an enabled autodiff context")
                };
                float_to_device_arms!(
                    @autodiff
                    *kind, &**device, ckp, $to_device;
                    $([$B1, $src_cfg]);*
                )

            }
            // Capture is deliberately absent from the cross-backend matrix. Same-backend movement
            // remains available; CaptureBackend decides whether the particular device transfer is
            // valid (computed tensors can only remain in their capture session).
            #[cfg(feature = "capture")]
            ($crate::DispatchTensorKind::Capture(kind), $crate::DispatchDevice::Capture(d)) => {
                $crate::DispatchTensor {
                    kind: $crate::DispatchTensorKind::Capture($crate::BackendTensor::Float(
                        $crate::backends::Capture::$to_device(kind.float(), d)
                    )),
                    autodiff: $tensor.autodiff,
                }
            }
            // --- Same backend to_device ---
            $(
                #[cfg($src_cfg)]
                ($crate::DispatchTensorKind::$B1(kind), $crate::DispatchDevice::$B1(d)) => {
                    $crate::DispatchTensor {
                        kind: $crate::DispatchTensorKind::$B1($crate::BackendTensor::Float(
                            $crate::backends::$B1::$to_device(kind.float(), d)
                        )),
                        autodiff: $tensor.autodiff,
                    }
                }
            )*

            // Materialize float tensors from any backend on capture.
            $(
                #[cfg(all($src_cfg, feature = "capture"))]
                ($crate::DispatchTensorKind::$B1(kind), $crate::DispatchDevice::Capture($device_ident)) => {
                    type B1 = $crate::backends::$B1;
                    type B2 = $crate::backends::Capture;
                    let $inner = kind.float();
                    $crate::DispatchTensor {
                        kind: $crate::DispatchTensorKind::Capture(
                            $crate::BackendTensor::Float($body)
                        ),
                        autodiff: $tensor.autodiff,
                    }
                }
            )*

            // --- Cross backend arms ---
            // This loop generates the grid of combinations
            $(
                $(
                    #[cfg(all($src_cfg, $dst_cfg))]
                    ($crate::DispatchTensorKind::$B1(kind), $crate::DispatchDevice::$B2($device_ident)) => {
                        type B1 = $crate::backends::$B1;
                        type B2 = $crate::backends::$B2;
                        let $inner = kind.float();

                        $crate::DispatchTensor {
                            kind: $crate::DispatchTensorKind::$B2($crate::BackendTensor::Float($body)),
                            autodiff: $tensor.autodiff,
                        }
                    }
                )+
            )*
            #[cfg(feature = "autodiff")]
            ($crate::DispatchTensorKind::Autodiff(..), _) | (_, $crate::DispatchDevice::Autodiff(_)) => panic!("Cannot move between autodiff and non-autodiff instances."),
            // Capture is intentionally one-way: initialized values can be moved onto a
            // capture device, but captured tensors have no materialized data to move back.
            #[cfg(feature = "capture")]
            ($crate::DispatchTensorKind::Capture(_), _) => {
                panic!("Cannot move a tensor from a capture device")
            }
        }
    };

    // Autodiff(DispatchTensor)
    (
        @autodiff
        $tensor:expr, $device:expr, $ckp:expr, $to_device:ident;
        $( [$B1:ident, $src_cfg:meta] );*
    ) => {{
        match ($tensor, $device) {
            // --- Same backend to_device ---
            $(
                #[cfg($src_cfg)]
                ($crate::DispatchTensorKind::$B1(tensor), $crate::DispatchDevice::$B1(d)) => {
                    let kind = $crate::DispatchTensorKind::Autodiff(alloc::boxed::Box::new($crate::DispatchTensorKind::$B1($crate::BackendTensor::Autodiff(
                        with_autodiff_backend!($B1, $ckp, |B| {
                            B::$to_device(tensor.autodiff(), d)
                        })
                    ))));
                    $crate::DispatchTensor {
                        kind,
                        autodiff: $crate::DispatchAutodiffContext::Enabled($ckp),
                    }
                }
            )*
            // TODO: should be possible
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

/// Unwraps a `Vec<DispatchTensor>` for a known backend.
macro_rules! unwrap_vec {
    ($Backend:ident, $vec:expr, $kind:ident) => {
        $vec.into_iter()
            .map(|t| match t.kind {
                $crate::DispatchTensorKind::$Backend(inner) => inner.$kind(),
                #[allow(unreachable_patterns)]
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
            .map(|t| match t.kind {
                $crate::DispatchTensorKind::Autodiff(inner) => match *inner {
                    $crate::DispatchTensorKind::$Backend(inner) => inner.$kind(),
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

/// Match arm generator for `transaction_op`.
macro_rules! transaction_op_arms {
    ($tx:ident, $first:expr; $([$Backend:ident, $cfg:meta]),*) => {{
        match &$first.kind {
            // Autodiff arm first
            #[cfg(feature = "autodiff")]
            $crate::DispatchTensorKind::Autodiff(inner) => {
                // Recursively dispatch on inner
                match **inner {
                    $(
                    #[cfg($cfg)]
                    $crate::DispatchTensorKind::$Backend(_) => {
                        type B = $crate::backends::$Backend;

                        // Unwrap vec
                        let floats = unwrap_vec!(@autodiff $Backend, $tx.read_floats, autodiff_inner);
                        let ints = unwrap_vec!($Backend, $tx.read_ints, int);
                        let bools = unwrap_vec!($Backend, $tx.read_bools, bool);
                        // Not supported
                        let qfloats = $tx.read_qfloats.into_iter().map(|_t| todo!("Quantization not supported yet")).collect();

                        B::tr_execute(TransactionPrimitive::new(floats, qfloats, ints, bools)).await
                    }
                )*
                    $crate::DispatchTensorKind::Autodiff(..) => unreachable!("Autodiff should not wrap an autodiff tensor.")
                }
            },

            $(
                #[cfg($cfg)]
                $crate::DispatchTensorKind::$Backend(_) => {
                    type B = $crate::backends::$Backend;

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
    }};
}

/// Helper to dispatch a transaction based on the first available tensor.
macro_rules! transaction_op {
    ($tx:ident, $first:expr) => {
        backend_list!(transaction_op_arms, $tx, $first)
    };
}
