use crate::{
    CubeRuntime,
    kernel::{
        bool_cast, into_contiguous, slice,
        utils::{address_type, decompose_linear, shape_divmod},
    },
    ops::{
        base::{into_data, reshape},
        numeric::{cumsum, empty_device_contiguous_dtype},
    },
    tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{DType, IntDType, Shape, TensorMetadata};
use cubecl::{CubeDim, calculate_cube_count_elemwise};
use cubecl::{prelude::*, std::FastDivmod, std::tensor::layout::linear::LinearView};

/// Scatter the coordinate of every `true` element to its output row.
///
/// One thread per input element. `offsets` is the inclusive prefix sum of the 0/1 mask, so for a
/// `true` element the running count `offsets[pos]` gives its 1-based row in the output; `false`
/// elements write nothing. `row * ndims + d` indexes the contiguous `[count, ndims]` output.
#[cube(launch_unchecked, address_type = "dynamic")]
fn argwhere_kernel<I: Int>(
    mask: LinearView<'_, I>,
    offsets: LinearView<'_, I>,
    output: &mut Tensor<I>,
    shape: Sequence<FastDivmod<usize>>,
    working_units: usize,
    #[comptime] ndims: usize,
    #[define(I)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }

    if usize::cast_from(mask.read(ABSOLUTE_POS)) != 0 {
        let row = usize::cast_from(offsets.read(ABSOLUTE_POS)) - 1;
        let (_, coords) = decompose_linear(ABSOLUTE_POS, &shape);
        #[unroll]
        for d in 0..ndims {
            output[row * ndims + d] = I::cast_from(*coords.index(d));
        }
    }
}

/// Compute the coordinates of the `true` elements, one row per element (row-major order).
///
/// GPU stream compaction: cast the mask to 0/1, prefix-sum it to get each element's output row,
/// read the total count back to the host to size the `[count, ndims]` output, then scatter each
/// `true` element's coordinate into its row. The count read is the only host synchronization; it
/// is data-dependent, which is why the op is asynchronous.
pub(crate) async fn argwhere<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    out_dtype: IntDType,
) -> CubeTensor<R> {
    let client = tensor.client.clone();
    let device = tensor.device.clone();
    let dtype: DType = out_dtype.into();

    // Flat position must map to a coordinate, so the mask has to be contiguous.
    let tensor = into_contiguous(tensor);
    let shape = tensor.shape();
    let ndims = shape.num_dims();
    let n = shape.num_elements();

    // 1. bool -> 0/1 int mask, keeping the original (contiguous) shape for coordinate decomposition.
    let mask = bool_cast(tensor, dtype);
    let coord_shape = shape_divmod(&mask);

    // 2. inclusive prefix sum over the flattened mask -> per-element output rows.
    let mask_flat = reshape(mask.clone(), Shape::new([n]));
    let offsets = cumsum(mask_flat.clone(), 0);

    // 3. total count = last prefix-sum element, read back to the host (the only sync).
    let count = if n == 0 {
        0
    } else {
        let last_row = (n - 1)..n;
        let last = slice(offsets.clone(), &[last_row]);
        into_data(last)
            .await
            .expect("Can read the count without error")
            .iter::<i64>()
            .next()
            .expect("The count slice has one element") as usize
    };

    // 4. empty result: no rows to scatter.
    if count == 0 {
        return empty_device_contiguous_dtype(client, device, Shape::new([0, ndims]), dtype);
    }

    // 5. allocate the [count, ndims] output.
    let output =
        empty_device_contiguous_dtype(client.clone(), device, Shape::new([count, ndims]), dtype);

    // 6. scatter each true element's coordinate to its row.
    let working_units = n;
    let cube_dim = CubeDim::new(&client, working_units);
    let cube_count = calculate_cube_count_elemwise(&client, working_units, cube_dim);
    unsafe {
        argwhere_kernel::launch_unchecked::<R>(
            &client,
            cube_count,
            cube_dim,
            address_type!(mask_flat, offsets, output),
            mask_flat.into_linear_view(),
            offsets.into_linear_view(),
            output.clone().into_tensor_arg(),
            coord_shape,
            working_units,
            ndims,
            dtype_to_storage_type(dtype),
        );
    }

    output
}
