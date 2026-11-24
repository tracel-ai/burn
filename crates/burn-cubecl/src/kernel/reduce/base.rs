#[cfg(feature = "autotune")]
use super::autotune_reduce_dims_plan;
use crate::ops::permute;
use crate::{
    ops::numeric::{empty_device_contiguous_dtype, fill_device_dtype, zeros_client},
    tensor::CubeTensor,
};
use burn_backend::cubecl::{dtype_to_elem_type, dtype_to_storage_type, elem_type_to_dtype};
use burn_backend::{DType, TensorMetadata};
use burn_std::{BoolDType, Metadata};
use burn_std::{Shape, Strides};
use cubecl::bytes::Bytes;
use cubecl::client::Client;
use cubecl::prelude::InputScalar;
use cubek::reduce::{
    ReduceError, ReduceOperation,
    kernels::{
        ReduceElements, ReduceKernel, ReduceKernelHeuristic, ReduceOpStrategy, ReduceWorkspace,
    },
};

/// The value a reduction over zero elements must produce, or `None` if there is none.
///
/// Reducing zero elements yields the folded operation's identity. The extrema have no identity in
/// a bounded numeric type (there is no integer below `i32::MIN`) and an `Arg*` would have to name
/// an element that does not exist, so they return `None`: numpy raises `ValueError` and torch
/// raises `IndexError` for all of them, as do burn's CPU backends for an empty `max`/`min`.
fn empty_reduce_identity(config: ReduceOperation, dtype: DType) -> Option<f64> {
    match config {
        ReduceOperation::Sum | ReduceOperation::Any => Some(0.0),
        ReduceOperation::Prod | ReduceOperation::All => Some(1.0),
        // Only floats can carry `NaN`; an integer mean of nothing has no representable value.
        ReduceOperation::Mean => dtype.is_float().then_some(f64::NAN),
        ReduceOperation::Max
        | ReduceOperation::Min
        | ReduceOperation::MaxAbs
        | ReduceOperation::TopK { .. }
        | ReduceOperation::ArgMax
        | ReduceOperation::ArgMin
        | ReduceOperation::ArgTopK { .. } => None,
    }
}

/// Fill `output` with the identity of `config`, or report that `config` has none.
///
/// Filled directly rather than by a reduce kernel because cubek's `validate_shapes` rejects a
/// zero-length axis with [`ReduceError::ReduceAxisTooSmall`], so no reduction can run.
///
/// An operation with no identity is rejected even when `output` is itself empty: emptiness of the
/// output depends on the *other* axes, so allowing it would make `max` succeed for shape `[0, 0]`
/// and fail for `[3, 0]`.
fn reduce_empty_axis(
    output: CubeTensor,
    axis_length: usize,
    config: ReduceOperation,
) -> Result<CubeTensor, ReduceError> {
    let identity =
        empty_reduce_identity(config, output.dtype).ok_or(ReduceError::ReduceAxisTooSmall {
            axis_length,
            k: accumulator_len(config),
        })?;

    if output.meta.num_elements() == 0 {
        return Ok(output);
    }

    let identity = InputScalar::new(identity, dtype_to_storage_type(output.dtype));

    Ok(fill_device_dtype(output, identity))
}

/// The sum of every element of `tensor`, in a tensor of shape `[1]`: one reduction over the
/// elements as one row where they lie contiguous, which cubek splits across cubes, and one a dim
/// otherwise.
pub fn sum(tensor: CubeTensor, strategy: KernelReduceStrategy) -> Result<CubeTensor, ReduceError> {
    // No kernel launches over an empty input, so write the additive identity.
    if tensor.meta.num_elements() == 0 {
        return Ok(zeros_client(
            tensor.client.clone(),
            tensor.device.clone(),
            [1].into(),
            tensor.dtype,
        ));
    }
    let mut tensor = crate::kernel::untile(tensor);
    if tensor.is_contiguous() {
        *tensor.meta = Metadata::new([tensor.meta.num_elements()], [1]);
    }
    reduce(tensor, None, strategy, ReduceOperation::Sum)
}

/// Reduce all elements of the `input` tensor using the instruction `Rd` and the given [strategy](KernelReduceStrategy).
///
/// Return an error if `strategy` is `Specific(strategy)` and the specified strategy is not supported by the `client`.
///
/// If there is no error, the output is a tensor with decreasing strides
/// where the shape of reduced dim is set to 1 but all shape are similar to the input.
pub fn reduce(
    mut tensor: CubeTensor,
    output_dtype: Option<DType>,
    strategy: KernelReduceStrategy,
    config: ReduceOperation,
) -> Result<CubeTensor, cubek::reduce::ReduceError> {
    // In practice, it looks like starting by the axis with the smallest shape
    // and going in increasing order lead to the fastest calculation.
    let sorted_axis = argsort(tensor.meta.shape());
    for axis in sorted_axis {
        tensor = reduce_dim(tensor, output_dtype, axis, strategy.clone(), config)?;
    }
    // reshape to scalar tensor
    *tensor.meta = Metadata::new([1], [1]);
    Ok(tensor)
}

/// Reduce several `dims` of the `input` tensor with the instruction `config`,
/// keeping each of them with length one.
///
/// Reducing one dimension at a time writes and reads back an intermediate per
/// dimension, the first of which is nearly the size of the input. Dimensions
/// that sit next to each other in memory can instead be folded into one axis by
/// a stride change alone, and reduced in a single launch — for a channels-last
/// tensor that is every non-channel dimension at once.
///
/// Only a run that memory already holds together folds, so each pass takes the
/// largest such run and reduces that, largest first because it is the pass that
/// leaves the least behind. The folded axis is presented *first* and the
/// remaining dimensions after it in logical order, which puts the reduction's
/// output back in logical order and contiguous, so the next pass starts from a
/// tensor whose memory order is its logical one. A layout holding every reduced
/// dimension together therefore takes a single launch, and one that scatters them
/// takes no more launches than reducing them one at a time would have.
/// With autotuning enabled, multi-axis sum and mean also compare this plan with
/// a permutation and reshape followed by one reduction, including any required copy.
pub fn reduce_dims(
    input: CubeTensor,
    output_dtype: Option<DType>,
    dims: &[usize],
    strategy: KernelReduceStrategy,
    config: ReduceOperation,
) -> Result<CubeTensor, ReduceError> {
    let input = crate::kernel::untile(input);

    #[cfg(feature = "autotune")]
    if matches!(strategy, KernelReduceStrategy::Autotune)
        && output_dtype.is_none()
        && matches!(config, ReduceOperation::Sum | ReduceOperation::Mean)
        && (matches!(config, ReduceOperation::Sum) || input.dtype.is_float())
        && (0..input.meta.num_dims())
            .filter(|dim| dims.contains(dim))
            .count()
            > 1
        && input.meta.shape().iter().all(|&size| size > 0)
    {
        return Ok(autotune_reduce_dims_plan(input, dims, config));
    }

    reduce_dims_folded(input, output_dtype, dims, strategy, config)
}

/// Reduces a rank-N input in contiguous memory runs, keeping reduced axes at size one.
pub(super) fn reduce_dims_folded(
    input: CubeTensor,
    output_dtype: Option<DType>,
    dims: &[usize],
    strategy: KernelReduceStrategy,
    config: ReduceOperation,
) -> Result<CubeTensor, ReduceError> {
    let rank = input.meta.num_dims();
    let shape = input.meta.shape();
    let reduced: Vec<usize> = (0..rank).filter(|dim| dims.contains(dim)).collect();

    let empty: Vec<usize> = reduced
        .iter()
        .copied()
        .filter(|dim| shape[*dim] == 0)
        .collect();

    // A dimension already of length one is reduced by being left alone, and its
    // stride is arbitrary, so keeping it among the dimensions to fold would let
    // it break a run of dimensions that do fold.
    let mut left: Vec<usize> = reduced
        .iter()
        .copied()
        .filter(|dim| shape[*dim] > 1)
        .collect();

    if empty.is_empty() {
        match (reduced.first(), left.len()) {
            (None, _) => return Ok(input),
            (Some(&dim), 0) => return reduce_dim(input, output_dtype, dim, strategy, config),
            (_, 1) => return reduce_dim(input, output_dtype, left[0], strategy, config),
            _ => {}
        }
    }

    let mut tensor = input;

    for dim in empty {
        tensor = reduce_dim(tensor, output_dtype, dim, strategy.clone(), config)?;
    }

    while !left.is_empty() {
        let run =
            largest_run_memory_holds_together(tensor.meta.shape(), tensor.meta.strides(), &left);
        tensor = reduce_folded_run(tensor, &run, output_dtype, strategy.clone(), config)?;

        left.retain(|dim| !run.contains(dim));
    }

    Ok(tensor)
}

fn reduce_folded_run(
    input: CubeTensor,
    run: &[usize],
    output_dtype: Option<DType>,
    strategy: KernelReduceStrategy,
    config: ReduceOperation,
) -> Result<CubeTensor, ReduceError> {
    let mut shape = input.meta.shape().clone();
    let rest = (0..shape.num_dims()).filter(|dim| !run.contains(dim));
    let order: Vec<_> = run.iter().copied().chain(rest).collect();
    let mut presented = permute(input, &order);
    fold_leading_dims(&mut presented, run.len());
    let mut output = reduce_dim(presented, output_dtype, 0, strategy, config)?;

    // The surviving axes were presented in logical order and the reduction writes contiguously.
    for &dim in run {
        shape[dim] = 1;
    }
    *output.meta = Metadata::new(shape.clone(), burn_std::tensor::contiguous_strides(&shape));
    Ok(output)
}

/// Sums a nonempty tensor over nonempty `dims` without narrowing any intermediate sums.
pub(crate) fn sum_dims_single_pass(
    input: CubeTensor,
    dims: &[usize],
) -> Result<CubeTensor, ReduceError> {
    let input = crate::kernel::untile(input);
    let shape = input.meta.shape();
    let run = largest_run_memory_holds_together(shape, input.meta.strides(), dims);
    let active_dims = dims.iter().filter(|&&dim| shape[dim] > 1).count();

    if run.len() < active_dims {
        return reduce_dims_copy(input, dims, ReduceOperation::Sum);
    }
    if run.is_empty() {
        return reduce_dim(
            input,
            None,
            dims[0],
            Default::default(),
            ReduceOperation::Sum,
        );
    }
    reduce_folded_run(input, &run, None, Default::default(), ReduceOperation::Sum)
}

/// Presents the input as `[kept_elements, reduced_elements]`, copying when required.
/// Restores the input rank with reduced axes at size one after a single reduction.
pub(super) fn reduce_dims_copy(
    input: CubeTensor,
    dims: &[usize],
    config: ReduceOperation,
) -> Result<CubeTensor, ReduceError> {
    let rank = input.meta.num_dims();
    let original_shape = input.meta.shape().clone();
    let kept: Vec<usize> = (0..rank).filter(|dim| !dims.contains(dim)).collect();
    let reduced: Vec<usize> = (0..rank).filter(|dim| dims.contains(dim)).collect();
    let kept_len = kept.iter().map(|&dim| original_shape[dim]).product();
    let reduced_len = reduced.iter().map(|&dim| original_shape[dim]).product();
    let order: Vec<usize> = kept.into_iter().chain(reduced.iter().copied()).collect();

    let presented = permute(input, &order);
    let folded = crate::ops::reshape(presented, Shape::new([kept_len, reduced_len]));
    let mut output = reduce_dim(folded, None, 1, Default::default(), config)?;

    let mut output_shape = original_shape;
    for dim in reduced {
        output_shape[dim] = 1;
    }
    *output.meta = Metadata::new(
        output_shape.clone(),
        burn_std::tensor::contiguous_strides(&output_shape),
    );
    Ok(output)
}

/// The dimensions among `left` that memory already holds together — consecutive in
/// memory order and nesting densely, so folding them into one axis is a stride
/// change — taking the run of most elements where there is more than one.
fn largest_run_memory_holds_together(
    shape: &Shape,
    strides: &Strides,
    left: &[usize],
) -> Vec<usize> {
    let mut memory_order: Vec<usize> = (0..shape.num_dims()).collect();
    memory_order.sort_by(|a, b| strides[*b].cmp(&strides[*a]).then(a.cmp(b)));

    let elements = |run: &[usize]| run.iter().map(|dim| shape[*dim]).product::<usize>();
    let mut largest: Vec<usize> = Vec::new();
    let mut run: Vec<usize> = Vec::new();

    for dim in memory_order {
        if shape[dim] == 1 {
            continue;
        }
        if !left.contains(&dim) {
            run.clear();
            continue;
        }
        // A gap under the dimension outside this one leaves the two spanning more
        // than their extents, and no stride change folds them into one axis.
        if let Some(&outside) = run.last()
            && strides[outside] != strides[dim] * shape[dim]
        {
            run.clear();
        }
        run.push(dim);

        if elements(&run) > elements(&largest) {
            largest.clone_from(&run);
        }
    }

    largest
}

/// Fold the leading `count` dimensions of `tensor` into one.
///
/// Sound only for dimensions memory holds together, which is what
/// [largest_run_memory_holds_together] returns: they span exactly their extents,
/// so the axis replacing them runs at the stride of the innermost of them.
fn fold_leading_dims(tensor: &mut CubeTensor, count: usize) {
    let shape = tensor.meta.shape();
    let strides = tensor.meta.strides();

    let mut folded_shape: Vec<usize> = vec![shape[..count].iter().product()];
    folded_shape.extend_from_slice(&shape[count..]);

    let mut folded_strides: Vec<usize> = vec![strides[count - 1]];
    folded_strides.extend_from_slice(&strides[count..]);

    *tensor.meta = Metadata::new(Shape::from(folded_shape), Strides::new(&folded_strides));
}

/// Reduce with a logical instruction ([`Any`](ReduceOperation::Any) /
/// [`All`](ReduceOperation::All)) and return the result as a boolean tensor.
///
/// `Any` / `All` require the output dtype (like `Arg*` index outputs): the
/// kernel writes the `0/1` flags directly into the numeric backing of the
/// boolean storage (cubek has no bool elem), so the only step left here is the
/// kernel-free relabel to `Bool`.
///
/// `dim == None` reduces the whole tensor to a scalar; `Some(dim)` reduces a
/// single axis, keeping it with length 1.
pub fn reduce_logical(
    tensor: CubeTensor,
    dim: Option<usize>,
    config: ReduceOperation,
    out_dtype: BoolDType,
) -> CubeTensor {
    debug_assert!(
        matches!(config, ReduceOperation::Any | ReduceOperation::All),
        "reduce_logical only supports Any / All, got {config:?}"
    );
    let out_bool = DType::Bool(out_dtype);
    let backing = elem_type_to_dtype(dtype_to_elem_type(out_bool));

    let mut out = match dim {
        Some(d) => reduce_dim(tensor, Some(backing), d, Default::default(), config),
        None => reduce(tensor, Some(backing), Default::default(), config),
    }
    .expect("Any/All reduce on a valid axis cannot fail");

    out.dtype = out_bool; // same storage, relabel as Bool (no kernel)
    out
}

/// Accumulator slots one reduction needs: `k` for top-k, `1` for every other operation.
///
/// Shared memory scales with it, so a routine that fits at one length can overrun the
/// device limit at another. That makes it part of the fused autotune key as well as the
/// output length along the reduced axis.
pub(crate) fn accumulator_len(config: ReduceOperation) -> usize {
    match config {
        ReduceOperation::TopK { k } | ReduceOperation::ArgTopK { k } => k,
        _ => 1,
    }
}

fn argsort(shape: &[usize]) -> Vec<usize> {
    let mut indices = (0..shape.len()).collect::<Vec<_>>();
    indices.sort_by_key(|&i| &shape[i]);
    indices
}

/// Reduce the given `axis` of the `input` tensor using the instruction `Rd` and the given [strategy](KernelReduceStrategy).
///
/// Return an error if `strategy` is `Specific(strategy)` and the specified strategy is not supported by the `client`.
/// Also returns an error if the `axis` is larger than the `input` rank or if the shape of `output` is invalid.
///
/// If there is no error, the output is a tensor with decreasing strides
/// where the shape of reduced dim is set to 1 but all shape are similar to the input.
pub fn reduce_dim(
    input: CubeTensor,
    output_dtype: Option<DType>,
    dim: usize,
    strategy: KernelReduceStrategy,
    config: ReduceOperation,
) -> Result<CubeTensor, cubek::reduce::ReduceError> {
    let input = crate::kernel::untile(input);
    debug_assert!(
        !matches!(
            config,
            ReduceOperation::ArgMax
                | ReduceOperation::ArgMin
                | ReduceOperation::ArgTopK { .. }
                | ReduceOperation::Any
                | ReduceOperation::All
        ) || output_dtype.is_some(),
        "The `output_dtype` has to be `Some` when the `config` is `ArgMax`, `ArgMin`, `ArgTopK`, `Any` or `All`.
        "
    );

    let accumulator_len = accumulator_len(config);
    let elements = ReduceElements::for_operation(
        config,
        dtype_to_elem_type(input.dtype),
        output_dtype.map(dtype_to_elem_type),
    );
    let client = input.client.clone();
    let output = init_reduce_output(&input, dim, &elements, accumulator_len).ok_or(
        cubek::reduce::ReduceError::InvalidAxis {
            axis: dim,
            rank: input.meta.num_dims(),
        },
    )?;

    // `output` already carries the right shape here, with `dim` set to `accumulator_len`.
    let axis_length = input.meta.shape[dim];
    if axis_length == 0 {
        return reduce_empty_axis(output, axis_length, config);
    }
    // A zero-length axis other than `dim` leaves no output to compute, and cubek cannot size a
    // launch over zero output units. A `k` too large for the axis still goes to cubek to be rejected.
    if output.meta.num_elements() == 0 && axis_length >= accumulator_len {
        return Ok(output);
    }

    let device = input.device.clone();
    let workspace = reduce_workspace(&client, &input, dim, config, elements, false);
    cubek::reduce::kernels::Reduce::new(strategy.into())
        .with_workspace(workspace)
        .execute(
            &client,
            &device,
            input.binding(),
            output.clone().binding(),
            dim,
            config,
            elements,
        )
        .map(|_| output)
}

/// Reduce the given `axis` of `input`, returning the values **and** their indices from a
/// single kernel launch.
///
/// Running the value reduction and its `Arg*` counterpart separately walks the input twice
/// and discards half of each result, even though one reduction already computes both. The
/// reduce kernels are memory bound, so folding the two launches into one roughly halves
/// the work.
///
/// `config` must be an operation with a meaningful index (top-k, max, min); each `Arg*`
/// config is an alias of its value counterpart here, since both halves are written either
/// way. Any other operation returns [`ReduceError::IndicesUnsupported`]. Both outputs are
/// contiguous with the reduced `dim` set to `k` for top-k and `1` otherwise.
pub fn reduce_dim_with_indices(
    input: CubeTensor,
    indices_dtype: DType,
    dim: usize,
    strategy: KernelReduceStrategy,
    config: ReduceOperation,
) -> Result<(CubeTensor, CubeTensor), ReduceError> {
    let unsupported = |operation| ReduceError::IndicesUnsupported { operation };

    // Fold each `Arg*` onto its value counterpart: its elements would otherwise demand an
    // output dtype, which here only ever applies to the indices.
    let config = match config {
        ReduceOperation::ArgMax => ReduceOperation::Max,
        ReduceOperation::ArgMin => ReduceOperation::Min,
        ReduceOperation::ArgTopK { k } => ReduceOperation::TopK { k },
        ReduceOperation::Max | ReduceOperation::Min | ReduceOperation::TopK { .. } => config,
        ReduceOperation::Sum => return Err(unsupported("Sum")),
        ReduceOperation::Prod => return Err(unsupported("Prod")),
        ReduceOperation::Mean => return Err(unsupported("Mean")),
        ReduceOperation::MaxAbs => return Err(unsupported("MaxAbs")),
        ReduceOperation::Any => return Err(unsupported("Any")),
        ReduceOperation::All => return Err(unsupported("All")),
    };

    let out_len = accumulator_len(config);

    // These operations keep input, values and accumulation at the input dtype; the index dtype is
    // the caller's and is converted for free in the final output write.
    let elements = ReduceElements::for_operation(config, dtype_to_elem_type(input.dtype), None);

    let invalid_axis = || ReduceError::InvalidAxis {
        axis: dim,
        rank: input.meta.num_dims(),
    };

    let values = init_reduce_output_dtype(&input, dim, elem_type_to_dtype(elements.out), out_len)
        .ok_or_else(invalid_axis)?;
    let indices =
        init_reduce_output_dtype(&input, dim, indices_dtype, out_len).ok_or_else(invalid_axis)?;

    // Every `config` reaching this point is an extremum, so an empty axis leaves it with no value
    // to report and no index to name. Always rejected, including when the outputs are themselves
    // empty - see `reduce_empty_axis`.
    if input.meta.shape[dim] == 0 {
        return Err(ReduceError::ReduceAxisTooSmall {
            axis_length: 0,
            k: out_len,
        });
    }
    // Same as in `reduce_dim`: an empty output has nothing to compute.
    if values.meta.num_elements() == 0 && input.meta.shape[dim] >= out_len {
        return Ok((values, indices));
    }

    let client = input.client.clone();

    let device = input.device.clone();
    let workspace = reduce_workspace(&client, &input, dim, config, elements, true);
    cubek::reduce::kernels::Reduce::new(strategy.into())
        .with_workspace(workspace)
        .execute_with_indices(
            &client,
            &device,
            input.binding(),
            values.clone().binding(),
            indices.clone().binding(),
            dim,
            config,
            elements,
            dtype_to_elem_type(indices_dtype),
        )
        .map(|_| (values, indices))
}

/// The scratch cubek's reductions that split a row across cubes take for reducing `dim` of
/// `input` by `config` in `elements`, `with_indices` where the indices go beside the values: cubek
/// allocates none, so its caller does, sized for whichever kernel the strategy picks, its arrival
/// counters zero.
fn reduce_workspace(
    client: &Client,
    input: &CubeTensor,
    dim: usize,
    config: ReduceOperation,
    elements: ReduceElements,
    with_indices: bool,
) -> ReduceWorkspace {
    let bytes = ReduceWorkspace::bytes_for(
        client,
        input.meta.shape(),
        dim,
        config,
        elements,
        with_indices,
    );
    // A runtime may refuse an empty allocation: every buffer holds at least one counter.
    let at_least_one = |bytes: u64| (bytes as usize).max(size_of::<u32>());
    let counters = at_least_one(bytes.arrivals) / size_of::<u32>();
    ReduceWorkspace {
        partials: client.empty(at_least_one(bytes.partials)),
        partial_indices: client.empty(at_least_one(bytes.partial_indices)),
        arrivals: client.create(Bytes::from_elems(vec![0u32; counters])),
    }
}

/// Creates an empty output tensor with the proper shape and decreasing strides to reduce the given `axis` of `input`
/// or return `None` if `axis` is out-of-bound.
pub fn init_reduce_output(
    input: &CubeTensor,
    dim: usize,
    elements: &ReduceElements,
    accumulator_len: usize,
) -> Option<CubeTensor> {
    init_reduce_output_dtype(
        input,
        dim,
        elem_type_to_dtype(elements.out),
        accumulator_len,
    )
}

/// Like [`init_reduce_output`], but with the output dtype given directly rather than taken
/// from a [`ReduceElements`]. Needed when one reduce writes two outputs of different dtypes.
pub fn init_reduce_output_dtype(
    input: &CubeTensor,
    dim: usize,
    dtype: DType,
    accumulator_len: usize,
) -> Option<CubeTensor> {
    (dim < input.meta.num_dims()).then(|| {
        let mut shape_out = input.shape();
        shape_out[dim] = accumulator_len;
        empty_device_contiguous_dtype(input.client.clone(), input.device.clone(), shape_out, dtype)
    })
}

/// Select a strategy to perform a reduction.
/// How a reduction picks its kernel. Every way runs cubek's tile-DSL reduce routines.
#[derive(Clone, Debug)]
pub enum KernelReduceStrategy {
    /// The routine cubek's rules pick, without a tuning round.
    Unspecified,
    /// Exactly this kernel.
    Specific(ReduceKernel),
    /// The routine measured fastest for the problem on this device.
    #[cfg(feature = "autotune")]
    Autotune,
}

impl From<KernelReduceStrategy> for ReduceOpStrategy {
    fn from(strategy: KernelReduceStrategy) -> Self {
        match strategy {
            KernelReduceStrategy::Unspecified => {
                ReduceOpStrategy::Heuristic(ReduceKernelHeuristic::Tile)
            }
            KernelReduceStrategy::Specific(kernel) => ReduceOpStrategy::Specific(kernel),
            #[cfg(feature = "autotune")]
            KernelReduceStrategy::Autotune => ReduceOpStrategy::Autotune,
        }
    }
}

impl Default for KernelReduceStrategy {
    fn default() -> Self {
        #[cfg(feature = "autotune")]
        return Self::Autotune;

        #[cfg(not(feature = "autotune"))]
        return Self::Unspecified;
    }
}
