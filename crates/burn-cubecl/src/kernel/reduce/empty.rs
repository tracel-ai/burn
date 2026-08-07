use crate::{CubeRuntime, kernel::utils::address_type, tensor::CubeTensor};
use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::{CubeDim, calculate_cube_count_elemwise};
use cubecl::{prelude::*, std::tensor::layout::linear::LinearViewMut};
use cubek::reduce::{ReduceError, components::instructions::ReduceOperationConfig};

// Identity kinds selected on the host and baked into the fill kernel at comptime.
const KIND_ZERO: u32 = 0;
const KIND_ONE: u32 = 1;
const KIND_NAN: u32 = 2;
const KIND_TYPE_MIN: u32 = 3;
const KIND_TYPE_MAX: u32 = 4;

/// Fill every output element with a reduction's identity value.
///
/// The extrema identities mirror `cubek`'s `max_identity` / `min_identity`: the identity for a max
/// reduction is the type minimum and for a min reduction the type maximum. Floats have no infinity
/// literal on every target, so they are built from their IEEE-754 bits, exactly as `cubek` does.
#[cube(launch_unchecked, address_type = "dynamic")]
fn fill_reduce_identity_kernel<E: Numeric>(
    mut output: LinearViewMut<'_, E>,
    #[comptime] kind: u32,
    #[define(E)] _dtype: StorageType,
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    let elem_type = type_of::<E>();
    let value = if comptime!(kind == KIND_ZERO) {
        E::from_int(0)
    } else if comptime!(kind == KIND_ONE) {
        E::from_int(1)
    } else if comptime!(kind == KIND_NAN) {
        E::cast_from(f32::reinterpret(0x7fc0_0000u32))
    } else if comptime!(kind == KIND_TYPE_MAX) {
        if comptime!(elem_type.is_float()) {
            E::cast_from(f32::reinterpret(0x7f80_0000u32))
        } else {
            E::max_value()
        }
    } else {
        if comptime!(elem_type.is_float()) {
            E::cast_from(f32::reinterpret(0xff80_0000u32))
        } else {
            E::min_value()
        }
    };

    output.write(ABSOLUTE_POS, value);
}

/// Result of reducing over a zero-length axis: every output position is the reduction identity.
///
/// `output` is the already-allocated reduce output (the reduced axis has length 1). There is no
/// element to fold, so the kernels (which require an axis length of at least 1) are skipped and the
/// identity is written directly. Operations whose result is an index (`ArgMax`, `ArgMin`, `TopK`,
/// `ArgTopK`) have no answer over an empty axis and return an error instead.
pub(crate) fn reduce_empty_axis<R: CubeRuntime>(
    output: CubeTensor<R>,
    config: ReduceOperationConfig,
) -> Result<CubeTensor<R>, ReduceError> {
    let kind = match config {
        ReduceOperationConfig::Sum | ReduceOperationConfig::MaxAbs | ReduceOperationConfig::Any => {
            KIND_ZERO
        }
        ReduceOperationConfig::Prod | ReduceOperationConfig::All => KIND_ONE,
        ReduceOperationConfig::Mean => KIND_NAN,
        ReduceOperationConfig::Max => KIND_TYPE_MIN,
        ReduceOperationConfig::Min => KIND_TYPE_MAX,
        ReduceOperationConfig::ArgMax
        | ReduceOperationConfig::ArgMin
        | ReduceOperationConfig::ArgTopK(_)
        | ReduceOperationConfig::TopK(_) => {
            return Err(ReduceError::Validation {
                details: "reducing over a zero-length axis has no result for this operation",
            });
        }
    };

    let num_elems = output.meta.num_elements();
    if num_elems == 0 {
        // Another axis is empty too, so the output holds no elements: nothing to fill.
        return Ok(output);
    }

    let client = output.client.clone();
    let dtype = output.dtype;
    let cube_dim = CubeDim::new(&client, num_elems);
    let cube_count = calculate_cube_count_elemwise(&client, num_elems, cube_dim);
    unsafe {
        fill_reduce_identity_kernel::launch_unchecked::<R>(
            &client,
            cube_count,
            cube_dim,
            address_type!(output),
            output.clone().into_linear_view(),
            kind,
            dtype_to_storage_type(dtype),
        );
    }

    Ok(output)
}
