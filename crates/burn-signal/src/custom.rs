//! Shared wire representation for fused, captured, and remote FFT operations.

use alloc::vec;
use burn_core::backend::ir::{
    BackendIr, CustomOpIr, HandleContainer, ScalarIr, TensorId, TensorIr,
};

use crate::SignalOps;

pub(crate) const RFFT: &str = "signal::rfft";
pub(crate) const IRFFT: &str = "signal::irfft";

fn scalars(dim: usize, n: Option<usize>) -> alloc::vec::Vec<ScalarIr> {
    // Keep presence distinct from value so graph caching preserves None vs Some(n).
    vec![
        ScalarIr::UInt(dim as u64),
        ScalarIr::Bool(n.is_some()),
        ScalarIr::UInt(n.unwrap_or_default() as u64),
    ]
}

pub(crate) fn rfft(
    signal: TensorIr,
    dim: usize,
    n: Option<usize>,
    mut new_id: impl FnMut() -> TensorId,
) -> CustomOpIr {
    let mut shape = signal.shape.clone();
    shape[dim] = n.unwrap_or(shape[dim]) / 2 + 1;
    let outputs = [
        TensorIr::uninit(new_id(), shape.clone(), signal.dtype),
        TensorIr::uninit(new_id(), shape, signal.dtype),
    ];
    CustomOpIr::with_scalars(RFFT, &[signal], &outputs, scalars(dim, n))
}

pub(crate) fn irfft(
    real: TensorIr,
    imag: TensorIr,
    dim: usize,
    n: Option<usize>,
    mut new_id: impl FnMut() -> TensorId,
) -> CustomOpIr {
    let mut shape = real.shape.clone();
    shape[dim] = n.unwrap_or_else(|| (shape[dim] - 1) * 2);
    let output = TensorIr::uninit(new_id(), shape, real.dtype);
    CustomOpIr::with_scalars(IRFFT, &[real, imag], &[output], scalars(dim, n))
}

pub(crate) fn execute<B: BackendIr + SignalOps>(
    handles: &mut HandleContainer<B::Handle>,
    desc: &CustomOpIr,
) {
    let [
        ScalarIr::UInt(dim),
        ScalarIr::Bool(has_n),
        ScalarIr::UInt(n),
    ] = desc.scalars.as_slice()
    else {
        panic!("Invalid scalar arguments for signal FFT custom operation")
    };
    let dim = *dim as usize;
    let n = has_n.then_some(*n as usize);
    match desc.id.as_str() {
        RFFT => {
            let ([input], [real_ir, imag_ir]) = desc.as_fixed();
            let input = handles.get_float_tensor::<B>(input);
            let (real, imag) = B::rfft(input, dim, n);
            handles.register_float_tensor::<B>(&real_ir.id, real);
            handles.register_float_tensor::<B>(&imag_ir.id, imag);
        }
        IRFFT => {
            let ([real_ir, imag_ir], [output]) = desc.as_fixed();
            let real = handles.get_float_tensor::<B>(real_ir);
            let imag = handles.get_float_tensor::<B>(imag_ir);
            let signal = B::irfft(real, imag, dim, n);
            handles.register_float_tensor::<B>(&output.id, signal);
        }
        id => panic!("Unknown signal FFT custom operation: {id}"),
    }
}

#[cfg(all(test, feature = "router", feature = "flex"))]
mod tests {
    use super::*;
    use burn_core::{
        backend::{
            Flex,
            ir::{OperationIr, TensorStatus},
        },
        tensor::{TensorData, Tolerance},
    };
    use burn_router::{CustomOpRegistry, TensorInterpreter};
    use burn_std::reader::try_read_sync;

    #[test]
    fn registered_fft_handlers_round_trip_different_axes_and_lengths() {
        let mut registry = CustomOpRegistry::<Flex>::new();
        crate::register_fft_ops(&mut registry);
        let mut interpreter = TensorInterpreter::with_custom_ops(Default::default(), registry);
        let mut id = 1_000_000;
        // Exercise both outputs and both scalars, including None vs an explicit natural length.
        for (dim, n) in [
            (0, None),
            (1, None),
            (1, Some(2)),
            (1, Some(4)),
            (1, Some(8)),
        ] {
            let input = interpreter.register_tensor_data_desc(TensorData::from([
                [1.0f32, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ]));
            let mut new_id = || {
                id += 1;
                TensorId::new(id)
            };
            let forward = rfft(input, dim, n, &mut new_id);
            let mut real = forward.outputs[0].clone();
            let mut imag = forward.outputs[1].clone();
            interpreter.register_op(OperationIr::Custom(forward));
            real.status = TensorStatus::ReadWrite;
            imag.status = TensorStatus::ReadWrite;
            let inverse = irfft(real, imag, dim, n, &mut new_id);
            let mut output = inverse.outputs[0].clone();
            interpreter.register_op(OperationIr::Custom(inverse));
            output.status = TensorStatus::ReadWrite;
            let data = try_read_sync(interpreter.read_tensor_async(output))
                .unwrap()
                .unwrap();
            let width = if dim == 1 { n.unwrap_or(4) } else { 4 };
            let expected: alloc::vec::Vec<f32> = (0..2)
                .flat_map(|row| {
                    (0..width).map(move |col| {
                        if col < 4 {
                            (row * 4 + col + 1) as f32
                        } else {
                            0.0
                        }
                    })
                })
                .collect();
            data.assert_approx_eq::<f32>(
                &TensorData::new(expected, [2, width]),
                Tolerance::absolute(1e-5),
            );
        }
    }

    #[test]
    fn fft_cache_key_preserves_optional_length() {
        use burn_core::tensor::{DType, Shape};
        let input = TensorIr::uninit(TensorId::new(0), Shape::from([4]), DType::F32);
        let implicit = rfft(input.clone(), 0, None, || TensorId::new(1));
        let explicit = rfft(input, 0, Some(4), || TensorId::new(1));
        assert_eq!(implicit.outputs, explicit.outputs);
        assert_ne!(implicit.scalars, explicit.scalars);
    }
}
