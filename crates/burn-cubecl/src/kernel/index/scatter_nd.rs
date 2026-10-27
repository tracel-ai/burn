use crate::{
    kernel::{
        AddOp, AssignOp, BinaryMaxOp, BinaryMinOp, BinaryOp, BinaryOpFamily, MulOp,
        utils::{address_type, shape_divmod, shape_divmod_range},
    },
    ops::numeric::zeros_client,
    tensor::CubeTensor,
};
use burn_backend::cubecl::{dtype_to_elem_type, dtype_to_storage_type};
use burn_backend::tensor::IndexingUpdateOp;
use burn_std::DType;
use cubecl::std::tensor::layout::linear::LinearView;
use cubecl::{CubeDim, calculate_cube_count_elemwise, features::AtomicUsage, ir::Type};
use cubecl::{prelude::*, std::FastDivmod};

/// Offsets of update element `pos`: where it lands in `target` (through the target's
/// strides) and where it is read in `values`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn scatter_nd_offsets<D: CubePrimitive, T: CubePrimitive, I: Int>(
    pos: usize,
    target: &mut Tensor<D>,
    indices: &LinearView<'_, I>,
    values: &Tensor<T>,
    data_slice_shape: &Sequence<FastDivmod<usize>>,
    values_shape: &Sequence<FastDivmod<usize>>,
    slice_size: usize,
    k: usize,
) -> (usize, usize) {
    let slice_offset = pos % slice_size;
    let update_idx = pos / slice_size;

    let idx_base = update_idx * k;
    let mut base_offset = 0usize;
    for j in 0..k {
        let idx_val = usize::cast_from(indices.read(idx_base + j));
        base_offset += idx_val * target.stride(j);
    }

    // Decompose slice_offset over data's trailing dims (k..n)
    let slice_rank = data_slice_shape.len().comptime();
    let mut data_slice_offset = 0usize;
    let mut remainder = slice_offset;
    #[unroll]
    for i in 0..slice_rank {
        let dim = slice_rank - i - 1;
        let (rem, coord) = data_slice_shape[dim].div_mod(remainder);
        remainder = rem;
        data_slice_offset += coord * target.stride(k + dim);
    }

    // values_shape covers all values dims (0..values_rank).
    let values_rank = values_shape.len().comptime();
    let batch_rank = values_rank - slice_rank;

    let mut val_offset = 0usize;
    let mut remainder_u = update_idx;
    #[unroll]
    for i in 0..batch_rank {
        let dim = batch_rank - i - 1;
        let (rem, coord) = values_shape[dim].div_mod(remainder_u);
        remainder_u = rem;
        val_offset += coord * values.stride(dim);
    }

    let mut remainder_v = slice_offset;
    #[unroll]
    for i in 0..slice_rank {
        let dim = slice_rank - i - 1;
        let val_dim = batch_rank + dim;
        let (rem, coord) = values_shape[val_dim].div_mod(remainder_v);
        remainder_v = rem;
        val_offset += coord * values.stride(val_dim);
    }

    (base_offset + data_slice_offset, val_offset)
}

/// scatter_nd GPU kernel.
///
/// Each thread handles one element across all update slices.
/// Work items = num_updates * slice_size.
#[cube(launch_unchecked, address_type = "dynamic")]
fn scatter_nd_kernel<T: Numeric, I: Int, Op: BinaryOpFamily>(
    data: &mut Tensor<T>,
    indices: LinearView<'_, I>,
    values: &Tensor<T>,
    data_slice_shape: Sequence<FastDivmod<usize>>,
    values_shape: Sequence<FastDivmod<usize>>,
    slice_size: usize,
    k: usize,
    working_units: usize,
    #[define(T, I)] _dtypes: [ElemType; 2],
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }

    let (data_idx, val_offset) = scatter_nd_offsets(
        ABSOLUTE_POS,
        data,
        &indices,
        values,
        &data_slice_shape,
        &values_shape,
        slice_size,
        k,
    );
    let result = Op::BinaryOp::<T, Const<1>>::execute(
        Vector::cast_from(data[data_idx]),
        Vector::cast_from(values[val_offset]),
    );
    data[data_idx] = result.extract(0usize);
}

/// Bool or/and scatter_nd: for each update equal to `absorbing` (true for or, false for
/// and), atomically stores `mark` at its position in `target`. Updates with the other value
/// leave the position unchanged, so repeated targets need no read-modify-write.
#[cube(launch_unchecked, address_type = "dynamic")]
fn bool_scatter_nd_mark_kernel<T: Numeric, M: Numeric, I: Int>(
    target: &mut Tensor<Atomic<M>>,
    indices: LinearView<'_, I>,
    values: &Tensor<T>,
    data_slice_shape: Sequence<FastDivmod<usize>>,
    values_shape: Sequence<FastDivmod<usize>>,
    slice_size: usize,
    k: usize,
    working_units: usize,
    #[comptime] absorbing: bool,
    #[comptime] mark: i64,
    #[define(T, M, I)] _dtypes: [ElemType; 3],
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }

    let (target_idx, val_offset) = scatter_nd_offsets(
        ABSOLUTE_POS,
        target,
        &indices,
        values,
        &data_slice_shape,
        &values_shape,
        slice_size,
        k,
    );
    let is_true = values[val_offset] != T::from_int(0);
    if is_true == absorbing {
        target[target_idx].store(M::from_int(mark));
    }
}

/// Writes `absorbing` into `data` wherever the contiguous `mask` (of `data`'s shape) is
/// set. One unit per element, so each position has a single writer.
#[cube(launch_unchecked, address_type = "dynamic")]
fn bool_scatter_nd_merge_kernel<T: Numeric, M: Numeric>(
    data: &mut Tensor<T>,
    mask: &Tensor<M>,
    data_shape: Sequence<FastDivmod<usize>>,
    num_elems: usize,
    #[comptime] absorbing: bool,
    #[define(T, M)] _dtypes: [ElemType; 2],
) {
    if ABSOLUTE_POS >= num_elems {
        terminate!();
    }
    if mask[ABSOLUTE_POS] == M::from_int(0) {
        terminate!();
    }

    let rank = data_shape.len().comptime();
    let mut remainder = ABSOLUTE_POS;
    let mut data_idx = 0usize;
    #[unroll]
    for i in 0..rank {
        let dim = rank - i - 1;
        let (rem, coord) = data_shape[dim].div_mod(remainder);
        remainder = rem;
        data_idx += coord * data.stride(dim);
    }
    data[data_idx] = T::from_int(absorbing as i64);
}

/// How the updates of a scatter_nd map onto `data`.
struct ScatterNdLayout {
    /// Number of leading data dims each index addresses.
    k: usize,
    /// Elements per update slice: the product of `data.shape[k..]`.
    slice_size: usize,
    /// One unit per update element: number of updates times `slice_size`.
    working_units: usize,
}

impl ScatterNdLayout {
    fn new(data: &CubeTensor, indices: &CubeTensor) -> Self {
        let data_shape = &data.meta.shape;
        let idx_shape = &indices.meta.shape;
        let m = idx_shape.num_dims();
        let k = idx_shape[m - 1];

        // num_updates = product of first M-1 dims of indices
        let num_updates: usize = idx_shape.as_slice()[..m - 1].iter().product();
        // slice_size = product of data.shape[K..]
        let slice_size: usize = data_shape.as_slice()[k..].iter().product();

        Self {
            k,
            slice_size,
            working_units: num_updates * slice_size,
        }
    }
}

/// The tensor itself when it can be written in place, otherwise a copy.
fn writable(tensor: CubeTensor) -> CubeTensor {
    match tensor.can_mut() && tensor.is_nonoverlapping() {
        true => tensor,
        false => tensor.copy(),
    }
}

pub(crate) fn scatter_nd(
    tensor: CubeTensor,
    indices: CubeTensor,
    values: CubeTensor,
    reduction: IndexingUpdateOp,
) -> CubeTensor {
    let tensor = writable(tensor);
    let layout = ScatterNdLayout::new(&tensor, &indices);

    let cube_dim = CubeDim::new(&indices.client, layout.working_units);
    let cube_count = calculate_cube_count_elemwise(&indices.client, layout.working_units, cube_dim);

    let (tensor_dtype, indices_dtype) = (tensor.dtype, indices.dtype);

    let launch = match reduction {
        IndexingUpdateOp::Assign => scatter_nd_kernel::launch_unchecked::<AssignOp>,
        IndexingUpdateOp::Add => scatter_nd_kernel::launch_unchecked::<AddOp>,
        IndexingUpdateOp::Mul => scatter_nd_kernel::launch_unchecked::<MulOp>,
        IndexingUpdateOp::Min => scatter_nd_kernel::launch_unchecked::<BinaryMinOp>,
        IndexingUpdateOp::Max => scatter_nd_kernel::launch_unchecked::<BinaryMaxOp>,
    };

    let data_slice_shape = shape_divmod_range(&tensor, layout.k..tensor.meta.shape.num_dims());
    let values_shape = shape_divmod_range(&values, 0..values.meta.shape.num_dims());

    unsafe {
        launch(
            &tensor.client.clone(),
            cube_count,
            cube_dim,
            address_type!(tensor, indices, values),
            tensor.clone().into_tensor_arg(),
            indices.into_linear_view(),
            values.into_tensor_arg(),
            data_slice_shape,
            values_shape,
            layout.slice_size,
            layout.k,
            layout.working_units,
            [
                dtype_to_storage_type(tensor_dtype),
                dtype_to_storage_type(indices_dtype),
            ],
        )
    }

    tensor
}

/// scatter_nd on bool data stored as 0/1: `Add`/`Max` are a logical or and `Mul`/`Min` a
/// logical and.
///
/// Each update can only set its position to the absorbing value (true for or, false for
/// and), so with repeated targets the updates are marked with atomic stores instead of a
/// read-modify-write. When the bool storage type has atomic stores, they go straight into
/// the data. Otherwise (u8 bools on most GPUs, which have no u8 atomics) they go into a
/// zeroed u32 mask, and a second pass writes the absorbing value where the mask is set.
pub(crate) fn bool_scatter_nd(
    tensor: CubeTensor,
    indices: CubeTensor,
    values: CubeTensor,
    reduction: IndexingUpdateOp,
) -> CubeTensor {
    let absorbing = match reduction {
        IndexingUpdateOp::Assign => return scatter_nd(tensor, indices, values, reduction),
        IndexingUpdateOp::Add | IndexingUpdateOp::Max => true,
        IndexingUpdateOp::Mul | IndexingUpdateOp::Min => false,
    };
    let tensor = writable(tensor);

    if stores_atomically(&tensor) {
        bool_scatter_nd_mark(tensor.clone(), indices, values, absorbing, absorbing as i64);
        return tensor;
    }

    let mask = zeros_client(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.meta.shape.clone(),
        DType::U32,
    );
    bool_scatter_nd_mark(mask.clone(), indices, values, absorbing, 1);
    bool_scatter_nd_merge(tensor, mask, absorbing)
}

/// Whether atomic loads and stores exist for the tensor's storage type.
fn stores_atomically(tensor: &CubeTensor) -> bool {
    tensor
        .client
        .properties()
        .atomic_type_usage(Type::atomic(dtype_to_elem_type(tensor.dtype)))
        .contains(AtomicUsage::LoadStore)
}

fn bool_scatter_nd_mark(
    target: CubeTensor,
    indices: CubeTensor,
    values: CubeTensor,
    absorbing: bool,
    mark: i64,
) {
    let layout = ScatterNdLayout::new(&target, &indices);

    let cube_dim = CubeDim::new(&indices.client, layout.working_units);
    let cube_count = calculate_cube_count_elemwise(&indices.client, layout.working_units, cube_dim);

    let dtypes = [
        dtype_to_storage_type(values.dtype),
        dtype_to_storage_type(target.dtype),
        dtype_to_storage_type(indices.dtype),
    ];
    let data_slice_shape = shape_divmod_range(&target, layout.k..target.meta.shape.num_dims());
    let values_shape = shape_divmod_range(&values, 0..values.meta.shape.num_dims());

    unsafe {
        bool_scatter_nd_mark_kernel::launch_unchecked(
            &target.client.clone(),
            cube_count,
            cube_dim,
            address_type!(target, indices, values),
            target.into_tensor_arg(),
            indices.into_linear_view(),
            values.into_tensor_arg(),
            data_slice_shape,
            values_shape,
            layout.slice_size,
            layout.k,
            layout.working_units,
            absorbing,
            mark,
            dtypes,
        )
    }
}

fn bool_scatter_nd_merge(tensor: CubeTensor, mask: CubeTensor, absorbing: bool) -> CubeTensor {
    let num_elems = tensor.meta.num_elements();
    let cube_dim = CubeDim::new(&tensor.client, num_elems);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, num_elems, cube_dim);
    let dtypes = [
        dtype_to_storage_type(tensor.dtype),
        dtype_to_storage_type(mask.dtype),
    ];

    unsafe {
        bool_scatter_nd_merge_kernel::launch_unchecked(
            &tensor.client.clone(),
            cube_count,
            cube_dim,
            address_type!(tensor, mask),
            tensor.clone().into_tensor_arg(),
            mask.into_tensor_arg(),
            shape_divmod(&tensor),
            num_elems,
            absorbing,
            dtypes,
        )
    }

    tensor
}
