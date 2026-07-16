use super::init_matmul_output;
use crate::{CubeRuntime, kernel::quantization::dequantize, tensor::CubeTensor};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{DType, TensorMetadata};
use burn_std::{MatmulTransformAnalysis, MatmulTransformPolicy, QuantLevel};
use cubecl::server::{GemmDescriptor, GemmMatrix};
use cubek::{
    matmul::{
        definition::{MatmulElems, MatmulGlobalElems, MatmulSetupError},
        strategy::Strategy,
    },
    std::InputBinding,
};

#[cfg(feature = "autotune")]
use super::matmul_autotune;

/// The strategy to be used when launching a matmul kernel.
pub enum MatmulStrategy {
    #[cfg(feature = "autotune")]
    /// Using autotune to choose the best kernel based on runtime information.
    Autotune,
    /// Cube implementation of matmul.
    Cube,
}

impl Default for MatmulStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "autotune")]
        return MatmulStrategy::Autotune;

        #[cfg(not(feature = "autotune"))]
        MatmulStrategy::Cube
    }
}

/// Launch a matmul kernel using the given strategy.
pub fn matmul<R: CubeRuntime>(
    mut lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: Option<CubeTensor<R>>,
    strategy: MatmulStrategy,
    out_dtype: DType,
) -> Result<CubeTensor<R>, MatmulSetupError> {
    let out = out.unwrap_or_else(|| init_matmul_output(&lhs, &rhs, out_dtype));

    // A broadcast-rhs batched matmul that would tile poorly is folded into a
    // single matmul: `[.., b, m, k] @ [.., 1, k, n]` runs as `[.., 1, b*m, k]`
    // instead of `b` matmuls that each re-read the whole rhs. Pure metadata —
    // the launch operands share the handles, the returned tensor keeps the
    // broadcast shape.
    let mut out_launch = out.clone();
    if lhs.qparams.is_none() {
        let analysis = MatmulTransformAnalysis::from_metadata(&lhs.meta, &rhs.meta, &out.meta);
        let action = MatmulTransformPolicy::default().action(&analysis);
        action.apply(&mut lhs.meta);
        action.apply(&mut out_launch.meta);
    }

    if lhs.qparams.is_none()
        && rhs.qparams.is_none()
        && try_accelerated_gemm(&lhs, &rhs, &out_launch).is_some()
    {
        return Ok(out);
    }

    match strategy {
        MatmulStrategy::Cube => {
            launch_matmul(&Default::default(), lhs, rhs, out_launch)?;
            Ok(out)
        }
        #[cfg(feature = "autotune")]
        MatmulStrategy::Autotune => {
            matmul_autotune(lhs, rhs, Some(out_launch), out_dtype);
            Ok(out)
        }
    }
}

pub(crate) fn launch_matmul_naive<R: CubeRuntime>(
    strategy: &Strategy,
    mut lhs: CubeTensor<R>,
    mut rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), MatmulSetupError> {
    // Naive has very specific layout requirements for block scaled tensors, so we need to manually
    // dequantize if it fails to launch normally. This is because naive is assumed to always work.
    if lhs.qparams.is_some() || rhs.qparams.is_some() {
        match launch_matmul(strategy, lhs.clone(), rhs.clone(), out.clone()) {
            Err(_) => {
                if lhs.qparams.is_some() {
                    lhs = dequantize(lhs, out.dtype);
                }
                if rhs.qparams.is_some() {
                    rhs = dequantize(rhs, out.dtype);
                }
                launch_matmul(strategy, lhs, rhs, out)
            }
            Ok(_) => Ok(()),
        }
    } else {
        launch_matmul(strategy, lhs, rhs, out)
    }
}

pub(crate) fn launch_matmul<R: CubeRuntime>(
    strategy: &Strategy,
    lhs: CubeTensor<R>,
    mut rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), MatmulSetupError> {
    let client = &out.client;

    let lhs_quant_handles = lhs.quantized_handles();
    let out_dtype: DType = out.dtype;

    let (lhs_dtype, lhs_handle) = match lhs_quant_handles {
        None => {
            let lhs_dtype = lhs.dtype;
            (
                lhs_dtype,
                InputBinding::new(lhs.binding(), dtype_to_storage_type(lhs_dtype)),
            )
        }
        Some((data, scale)) => {
            let scheme = lhs.scheme();
            let data_dtype = data.dtype;
            let scale_dtype = scale.dtype;
            (
                out_dtype,
                InputBinding::quantized(
                    data.binding(),
                    scale.binding(),
                    lhs.meta.shape().clone(),
                    scheme,
                    dtype_to_storage_type(data_dtype),
                    dtype_to_storage_type(scale_dtype),
                ),
            )
        }
    };

    let rhs_quant_handles = rhs.quantized_handles();

    let (rhs_dtype, rhs_handle) = match rhs_quant_handles {
        None => (
            lhs_dtype,
            InputBinding::new(rhs.binding(), dtype_to_storage_type(lhs_dtype)),
        ),
        Some((data, scale)) => {
            // Extremely hacky fix to ensure naive can run in every case
            if matches!(strategy, Strategy::Naive)
                && matches!(rhs.scheme().level, QuantLevel::Block(_))
            {
                rhs = dequantize(rhs.clone(), lhs_dtype);
                let rhs_dtype = rhs.dtype;
                (
                    lhs_dtype,
                    InputBinding::new(rhs.binding(), dtype_to_storage_type(rhs_dtype)),
                )
            } else {
                let scheme = rhs.scheme();
                let data_dtype = data.dtype;
                let scale_dtype = scale.dtype;
                (
                    out_dtype,
                    InputBinding::quantized(
                        data.binding(),
                        scale.binding(),
                        rhs.meta.shape().clone(),
                        scheme,
                        dtype_to_storage_type(data_dtype),
                        dtype_to_storage_type(scale_dtype),
                    ),
                )
            }
        }
    };

    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: dtype_to_storage_type(lhs_dtype),
        rhs: dtype_to_storage_type(rhs_dtype),
        out: dtype_to_storage_type(out_dtype),
    });

    cubek::matmul::launch::launch_ref(
        strategy,
        client,
        lhs_handle,
        rhs_handle,
        out.clone().binding(),
        &mut dtypes,
    )?;

    Ok(())
}

fn try_accelerated_gemm<R: CubeRuntime>(
    lhs: &CubeTensor<R>,
    rhs: &CubeTensor<R>,
    out: &CubeTensor<R>,
) -> Option<()> {
    if lhs.dtype != rhs.dtype || lhs.dtype != out.dtype {
        return None;
    }
    let elem = dtype_to_storage_type(lhs.dtype).elem_type();
    if !out
        .client
        .properties()
        .features
        .matmul
        .accelerated_gemm
        .contains(&elem)
    {
        return None;
    }

    let lhs_shape = lhs.meta.shape();
    let rhs_shape = rhs.meta.shape();
    let out_shape = out.meta.shape();
    let rank = lhs_shape.rank();
    if rank < 2 || rhs_shape.rank() != rank || out_shape.rank() != rank {
        return None;
    }

    let m = lhs_shape[rank - 2];
    let k = lhs_shape[rank - 1];
    let n = rhs_shape[rank - 1];
    if rhs_shape[rank - 2] != k || out_shape[rank - 2] != m || out_shape[rank - 1] != n {
        return None;
    }
    // The native primitive intentionally has no zero-fill side path for a
    // nonempty zero-K result. CubeK retains the complete tensor semantics.
    if k == 0 && m != 0 && n != 0 {
        return None;
    }

    let batch_shape = &out_shape[..rank - 2];
    let batch_count = batch_shape.iter().product::<usize>();
    let mut lhs_desc = matrix_descriptor(lhs, m, k, batch_shape)?;
    let rhs_desc = matrix_descriptor(rhs, k, n, batch_shape)?;
    let mut out_desc = matrix_descriptor(out, m, n, batch_shape)?;
    if out_desc.transposed {
        return None;
    }

    let mut launch_m = m;
    let mut launch_batches = batch_count;
    if batch_count > 1
        && rhs_desc.batch_stride == 0
        && !lhs_desc.transposed
        && lhs_desc.batch_stride == m.checked_mul(lhs_desc.leading_dimension as usize)? as u64
        && out_desc.batch_stride == m.checked_mul(out_desc.leading_dimension as usize)? as u64
    {
        launch_m = m.checked_mul(batch_count)?;
        launch_batches = 1;
        lhs_desc.batch_stride = 0;
        out_desc.batch_stride = 0;
    }

    let descriptor = GemmDescriptor::new(
        lhs_desc,
        rhs_desc,
        out_desc,
        launch_m.try_into().ok()?,
        n.try_into().ok()?,
        k.try_into().ok()?,
        launch_batches.try_into().ok()?,
        elem,
    );

    out.client.gemm(descriptor);
    Some(())
}

fn matrix_descriptor<R: CubeRuntime>(
    tensor: &CubeTensor<R>,
    rows: usize,
    cols: usize,
    batch_shape: &[usize],
) -> Option<GemmMatrix> {
    let rank = tensor.meta.rank();
    let shape = tensor.meta.shape();
    let strides = tensor.meta.strides();
    let row_stride = strides[rank - 2];
    let col_stride = strides[rank - 1];
    let (leading_dimension, transposed) = if col_stride == 1 && row_stride >= cols {
        (row_stride, false)
    } else if row_stride == 1 && col_stride >= rows {
        (col_stride, true)
    } else {
        return None;
    };

    let batch_stride = if batch_shape.is_empty() {
        0
    } else if shape[..rank - 2].iter().all(|dim| *dim == 1) {
        0
    } else {
        if shape[..rank - 2] != *batch_shape {
            return None;
        }
        for dim in 0..rank.saturating_sub(3) {
            if strides[dim]
                != shape[dim + 1]
                    .checked_mul(strides[dim + 1])
                    .unwrap_or(usize::MAX)
            {
                return None;
            }
        }
        strides[rank - 3] as u64
    };

    Some(GemmMatrix::new(
        tensor.handle.clone().binding(),
        leading_dimension.try_into().ok()?,
        batch_stride,
        transposed,
    ))
}
