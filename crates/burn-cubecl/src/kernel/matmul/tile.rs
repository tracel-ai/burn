//! Matmul over cubek's tile-DSL routines ([`cubek::matmul::kernels`]). cubek's selection plans the
//! product: under autotune, the few-rows or the many-rows autotune its rows go to; by rule otherwise.
//! A quantized input is read as stored by the routines listed for it or decoded first, and a scheme
//! none reads is dequantized here first.

use crate::{
    CubeTuneId,
    kernel::{quantization::dequantize, untile},
    ops::swap_dims,
    tensor::CubeTensor,
};
use burn_backend::DType;
use burn_backend::cubecl::dtype_to_storage_type;
use burn_std::Metadata;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::{Axis, launch::Refusal};
use cubek::matmul::{
    definition::MatmulSetupError,
    kernels::{
        MatmulOpStrategy,
        plan::{FrameworkDequantize, MatmulCall, MatmulCallInput},
        select::{
            problem::{MatmulProblem, OperandKind},
            selector::select,
        },
        tensor::StoredTensor,
        tune::MatmulAutotuneGroups,
    },
};
use cubek::std::{
    Hardware,
    matmul_axes::{K, M, N},
    plan::Plan,
};
use std::sync::Arc;

type Inputs = (CubeTensor, CubeTensor, CubeTensor);

/// Writes `lhs · rhs` into `out`, through the plan cubek's selection gives under `strategy`.
///
/// # Errors
///
/// A product cubek's selection refuses or no routine of its plan serves.
pub(crate) fn launch(
    lhs: CubeTensor,
    rhs: CubeTensor,
    out: CubeTensor,
    out_dtype: DType,
    strategy: MatmulOpStrategy,
) -> Result<(), MatmulSetupError> {
    if lhs.meta.shape().iter().any(|&d| d == 0) || rhs.meta.shape().iter().any(|&d| d == 0) {
        return Ok(());
    }

    let (lhs, rhs, out) = run_as((lhs, rhs, out));
    let hw = Hardware::from_client(&lhs.client);
    let (client, tune_id) = (
        lhs.client.clone(),
        CubeTuneId::new(&lhs.client, &lhs.device),
    );
    let lhs = call_input(lhs, [M, K], out_dtype).map_err(invalid)?;
    let rhs = call_input(rhs, [K, N], out_dtype).map_err(invalid)?;
    let problem = problem(
        (&lhs.meta, lhs.tensor.kind()),
        (&rhs.meta, rhs.tensor.kind()),
    )?;
    let plan = select(strategy, &problem, &hw, MatmulAutotuneGroups::default()).map_err(invalid)?;
    let call = MatmulCall {
        hw,
        client,
        tune_id,
        lhs,
        rhs,
        out: dtype_to_storage_type(out_dtype),
        destination: Some(handle_of(out)),
    };
    plan.launch(&call).map_err(invalid)?;
    Ok(())
}

/// cubek's view of `lhs · rhs`, from each operand's logical extents: a storage-tiled operand's
/// metadata shape is physical, its tiles counted as dims of their own.
fn problem(
    (lhs, lhs_kind): (&Metadata, OperandKind),
    (rhs, rhs_kind): (&Metadata, OperandKind),
) -> Result<MatmulProblem, MatmulSetupError> {
    let logical = |meta: &Metadata| {
        meta.logical_shape()
            .map_err(|why| invalid(format!("{why:?}")))
    };
    Ok(MatmulProblem::new(
        (&logical(lhs)?, lhs_kind),
        (&logical(rhs)?, rhs_kind),
    ))
}

/// What cubek answered, as the error burn's matmul reports.
fn invalid(why: impl std::fmt::Display + 'static) -> MatmulSetupError {
    MatmulSetupError::InvalidConfig(Box::new(why))
}

/// The product the routines run for `lhs · rhs` into `out`: itself, or, where it writes a single
/// column, its transpose `(rhsᵀ · lhsᵀ)ᵀ`, the same buffers viewed through swapped strides.
///
/// The routines cut the output's columns into blocks a cube or a unit covers, so a single column
/// leaves almost every lane of a block idle, and nothing reads the long lhs faster than its
/// rows; transposed, the product is one row against a long rhs, which the routines that hold
/// every row whole stream at the bus. The output of one column stays contiguous transposed; a
/// wider one would be written across its strides, so it runs as it is.
fn run_as((lhs, rhs, out): Inputs) -> Inputs {
    let rank = out.meta.num_dims();
    let (rows, columns) = (out.meta.shape()[rank - 2], out.meta.shape()[rank - 1]);
    if columns != 1 || rows == 1 {
        return (lhs, rhs, out);
    }
    let transposed = |tensor: CubeTensor| {
        let rank = tensor.meta.num_dims();
        swap_dims(tensor, rank - 2, rank - 1)
    };
    (transposed(rhs), transposed(lhs), transposed(out))
}

/// `tensor`, the input stated over `axes`, as cubek's plan reads it: as stored, and dequantized
/// into `out` by burn's dequantize where a plan reaches for it. A quantized input the routines
/// cannot state as stored is dequantized here first, and the plan reads a float input; a float
/// one stored in tiles no routine reads is laid back in rows.
///
/// # Errors
///
/// A float input stored in a way the routines cannot state.
fn call_input(tensor: CubeTensor, axes: [Axis; 2], out: DType) -> Result<MatmulCallInput, Refusal> {
    let quantized = tensor.try_scheme().is_some();
    let framework_dequantize = quantized.then(|| {
        let packed = tensor.clone();
        Arc::new(move || Ok(dequantize(packed.clone(), out).plain())) as FrameworkDequantize
    });
    let meta = (*tensor.meta).clone();
    match tensor.clone().stored(axes) {
        // No routine reads tiles stored a column at a time, as a transposed weight's are: such
        // an input is read as rows.
        Ok(StoredTensor::Plain(plain)) if plain.is_tiled_column_first() => {
            Ok(MatmulCallInput::plain(untile(tensor).plain()))
        }
        Ok(stored) => Ok(MatmulCallInput {
            tensor: stored,
            meta,
            framework_dequantize,
        }),
        Err(_) if quantized => Ok(MatmulCallInput::plain(dequantize(tensor, out).plain())),
        Err(why) => Err(why),
    }
}

/// `tensor`'s buffer under its shape and strides, the output a plan writes into.
fn handle_of(tensor: CubeTensor) -> TensorHandle {
    let shape = tensor.meta.shape().clone();
    let strides = tensor.meta.strides().clone();
    TensorHandle::new(
        tensor.handle,
        shape,
        strides,
        dtype_to_storage_type(tensor.dtype),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::zspace::Tiling;
    use cubek::matmul::kernels::tune::MatmulAutotune;

    /// Both autotunes compute a decode product alike, so only its routing shows the weight's
    /// logical shape reached cubek.
    #[test]
    fn a_decode_product_against_a_tiled_weight_goes_to_the_few_rows_autotune() {
        let input = Metadata::new([1, 256], [256, 1]);
        // `[256, 512]` in `[32, 64]` tiles.
        let weight = Metadata::new([8, 8, 32, 64], [16384, 2048, 64, 1])
            .with_tiling(Tiling::new(&[2, 2]).unwrap())
            .unwrap();
        let problem = problem((&input, OperandKind::Float), (&weight, OperandKind::Float)).unwrap();
        assert_eq!(problem.streamed_rows, Some(1));
        assert_eq!(MatmulAutotune::new(&problem), MatmulAutotune::FewRows);
    }
}
