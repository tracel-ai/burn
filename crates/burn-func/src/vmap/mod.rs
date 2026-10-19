use alloc::vec::Vec;
use burn_backend::{
    Backend, BackendTypes, BoolDType, DType, DTypeUsageSet, Distribution, ExecutionError,
    FloatDType, GraphUnsupported, IntDType, MemoryPoolUsage, ProfileDuration, ProfileOptions,
    ProfileToken, Scalar, Shape, Slice, SlicedPoolReport, TensorData, TensorMetadata,
    ops::FloatTensorOps,
    tensor::{BoolTensor, FloatTensor, IntTensor},
};
use burn_std::ops::PadMode;
use core::marker::PhantomData;

use crate::PyTree;

/// Module, integer, boolean and quantized delegation for `Vmap`.
pub mod module;

/// A primitive with an optional physical batch dimension at axis 0.
///
/// When `batch_size` is `Some(n)`, `shape()` and `rank()` describe the logical, unbatched
/// tensor: the physical primitive has shape `[n, ..logical_shape]`. This keeps the high-level
/// tensor API's shape assertions and rank arithmetic in terms of the function being mapped,
/// not the vectorized batch.
#[derive(Clone, Debug)]
pub struct BatchedPrimitive<P> {
    /// The physical primitive.
    pub primitive: P,
    /// `Some(n)` when physical axis 0 is a batch dimension of size `n`.
    pub batch_size: Option<usize>,
}

impl<P> BatchedPrimitive<P> {
    /// Wraps a physical primitive with a batch dimension.
    pub fn batched(primitive: P, batch_size: usize) -> Self {
        Self {
            primitive,
            batch_size: Some(batch_size),
        }
    }

    /// Wraps a physical primitive with no batch dimension.
    pub fn unbatched(primitive: P) -> Self {
        Self {
            primitive,
            batch_size: None,
        }
    }
}

impl<P: TensorMetadata> TensorMetadata for BatchedPrimitive<P> {
    type Device = P::Device;

    fn dtype(&self) -> DType {
        self.primitive.dtype()
    }

    fn shape(&self) -> Shape {
        let shape = self.primitive.shape();
        match self.batch_size {
            Some(_) => Shape::from(shape.as_slice()[1..].to_vec()),
            None => shape,
        }
    }

    fn rank(&self) -> usize {
        match self.batch_size {
            Some(_) => self.primitive.rank() - 1,
            None => self.primitive.rank(),
        }
    }

    fn device(&self) -> P::Device {
        self.primitive.device()
    }

    fn can_mut(&self) -> bool {
        self.primitive.can_mut()
    }
}

/// Vectorized batching as a backend decorator.
#[derive(Clone, Copy, Debug, Default)]
pub struct Vmap<B> {
    _b: PhantomData<B>,
}

impl<B: Backend> BackendTypes for Vmap<B> {
    type Device = B::Device;
    type FloatTensorPrimitive = BatchedPrimitive<B::FloatTensorPrimitive>;
    type IntTensorPrimitive = BatchedPrimitive<B::IntTensorPrimitive>;
    type BoolTensorPrimitive = BatchedPrimitive<B::BoolTensorPrimitive>;
    type QuantizedTensorPrimitive = B::QuantizedTensorPrimitive;
    type GraphPrimitive = GraphUnsupported;
}

impl<B: Backend> Backend for Vmap<B> {
    fn ad_enabled(device: &B::Device) -> bool {
        B::ad_enabled(device)
    }

    fn name(device: &B::Device) -> alloc::string::String {
        alloc::format!("vmap<{}>", B::name(device))
    }

    fn seed(device: &B::Device, seed: u64) {
        B::seed(device, seed)
    }

    fn sync(device: &B::Device) -> Result<(), ExecutionError> {
        B::sync(device)
    }

    fn profile<O: Send + 'static>(
        device: &B::Device,
        options: ProfileOptions,
        func: impl FnOnce() -> O + Send,
    ) -> Result<(O, ProfileDuration), ExecutionError> {
        B::profile(device, options, func)
    }

    fn profile_start(device: &B::Device) -> Result<Option<ProfileToken>, ExecutionError> {
        B::profile_start(device)
    }

    fn profile_end(
        device: &B::Device,
        token: ProfileToken,
        options: ProfileOptions,
    ) -> Result<ProfileDuration, ExecutionError> {
        B::profile_end(device, token, options)
    }

    fn profile_abandon(device: &B::Device, token: ProfileToken) {
        B::profile_abandon(device, token)
    }

    fn memory_persistent_allocations<
        Output: Send,
        Input: Send,
        Func: Fn(Input) -> Output + Send,
    >(
        device: &B::Device,
        input: Input,
        func: Func,
    ) -> Output {
        B::memory_persistent_allocations(device, input, func)
    }

    fn memory_cleanup(device: &B::Device) {
        B::memory_cleanup(device)
    }

    fn memory_pool_report(device: &B::Device) -> Option<Vec<SlicedPoolReport>> {
        B::memory_pool_report(device)
    }

    fn memory_pool_usage(device: &B::Device) -> Option<MemoryPoolUsage> {
        B::memory_pool_usage(device)
    }

    fn staging<'a, Iter>(data: Iter, device: &B::Device)
    where
        Iter: Iterator<Item = &'a mut TensorData>,
    {
        B::staging(data, device);
    }

    fn supports_dtype(device: &B::Device, dtype: DType) -> bool {
        B::supports_dtype(device, dtype)
    }

    fn dtype_usage(device: &B::Device, dtype: DType) -> DTypeUsageSet {
        B::dtype_usage(device, dtype)
    }

    fn device_count(type_id: u16) -> usize {
        B::device_count(type_id)
    }

    fn flush(device: &B::Device) -> Result<(), ExecutionError> {
        B::flush(device)
    }
}

type Batched<P> = BatchedPrimitive<P>;

pub(crate) fn promote_float<B: Backend>(
    tensor: BatchedPrimitive<FloatTensor<B>>,
    batch_size: usize,
) -> FloatTensor<B> {
    match tensor.batch_size {
        Some(existing) => {
            assert_eq!(
                existing, batch_size,
                "Vmap batch sizes must agree while promoting a tensor"
            );
            tensor.primitive
        }
        None => {
            let shape = tensor.primitive.shape();
            let logical = shape.as_slice();
            let mut reshape = Vec::with_capacity(logical.len() + 1);
            reshape.push(1);
            reshape.extend_from_slice(logical);
            let reshaped = B::float_reshape(tensor.primitive, Shape::from(reshape));
            let mut target = Vec::with_capacity(logical.len() + 1);
            target.push(batch_size);
            target.extend_from_slice(logical);
            B::float_expand(reshaped, Shape::from(target))
        }
    }
}

pub(crate) fn promote_int<B: Backend>(
    tensor: BatchedPrimitive<IntTensor<B>>,
    batch_size: usize,
) -> IntTensor<B> {
    match tensor.batch_size {
        Some(existing) => {
            assert_eq!(existing, batch_size);
            tensor.primitive
        }
        None => {
            let shape = tensor.primitive.shape();
            let logical = shape.as_slice();
            let mut reshape = Vec::with_capacity(logical.len() + 1);
            reshape.push(1);
            reshape.extend_from_slice(logical);
            let reshaped = B::int_reshape(tensor.primitive, Shape::from(reshape));
            let mut target = Vec::with_capacity(logical.len() + 1);
            target.push(batch_size);
            target.extend_from_slice(logical);
            B::int_expand(reshaped, Shape::from(target))
        }
    }
}

pub(crate) fn promote_bool<B: Backend>(
    tensor: BatchedPrimitive<BoolTensor<B>>,
    batch_size: usize,
) -> BoolTensor<B> {
    match tensor.batch_size {
        Some(existing) => {
            assert_eq!(existing, batch_size);
            tensor.primitive
        }
        None => {
            let shape = tensor.primitive.shape();
            let logical = shape.as_slice();
            let mut reshape = Vec::with_capacity(logical.len() + 1);
            reshape.push(1);
            reshape.extend_from_slice(logical);
            let reshaped = B::bool_reshape(tensor.primitive, Shape::from(reshape));
            let mut target = Vec::with_capacity(logical.len() + 1);
            target.push(batch_size);
            target.extend_from_slice(logical);
            B::bool_expand(reshaped, Shape::from(target))
        }
    }
}
fn align_ranks_post_axis_0<T>(
    mut lhs: T,
    mut rhs: T,
    reshape: impl Fn(T, Shape) -> T,
    shape_of: impl Fn(&T) -> Shape,
) -> (T, T) {
    let r_lhs = shape_of(&lhs).num_dims();
    let r_rhs = shape_of(&rhs).num_dims();
    if r_lhs < r_rhs {
        let diff = r_rhs - r_lhs;
        let mut dims = shape_of(&lhs).as_slice().to_vec();
        for _ in 0..diff {
            dims.insert(1, 1);
        }
        lhs = reshape(lhs, Shape::from(dims));
    } else if r_rhs < r_lhs {
        let diff = r_lhs - r_rhs;
        let mut dims = shape_of(&rhs).as_slice().to_vec();
        for _ in 0..diff {
            dims.insert(1, 1);
        }
        rhs = reshape(rhs, Shape::from(dims));
    }
    (lhs, rhs)
}

pub(crate) fn unify_float<B: Backend>(
    lhs: BatchedPrimitive<FloatTensor<B>>,
    rhs: BatchedPrimitive<FloatTensor<B>>,
) -> (FloatTensor<B>, FloatTensor<B>, Option<usize>) {
    let (lhs, rhs, batch_size) = match (lhs.batch_size, rhs.batch_size) {
        (None, None) => return (lhs.primitive, rhs.primitive, None),
        (Some(n), None) => (lhs.primitive, promote_float::<B>(rhs, n), Some(n)),
        (None, Some(n)) => (promote_float::<B>(lhs, n), rhs.primitive, Some(n)),
        (Some(a), Some(b)) => {
            assert_eq!(a, b, "Vmap batch sizes must agree");
            (lhs.primitive, rhs.primitive, Some(a))
        }
    };
    let (lhs, rhs) =
        align_ranks_post_axis_0(lhs, rhs, |t, s| B::float_reshape(t, s), |t| t.shape());
    (lhs, rhs, batch_size)
}

pub(crate) fn unify_int<B: Backend>(
    lhs: BatchedPrimitive<IntTensor<B>>,
    rhs: BatchedPrimitive<IntTensor<B>>,
) -> (IntTensor<B>, IntTensor<B>, Option<usize>) {
    let (lhs, rhs, batch_size) = match (lhs.batch_size, rhs.batch_size) {
        (None, None) => return (lhs.primitive, rhs.primitive, None),
        (Some(n), None) => (lhs.primitive, promote_int::<B>(rhs, n), Some(n)),
        (None, Some(n)) => (promote_int::<B>(lhs, n), rhs.primitive, Some(n)),
        (Some(a), Some(b)) => {
            assert_eq!(a, b, "Vmap batch sizes must agree");
            (lhs.primitive, rhs.primitive, Some(a))
        }
    };
    let (lhs, rhs) = align_ranks_post_axis_0(lhs, rhs, |t, s| B::int_reshape(t, s), |t| t.shape());
    (lhs, rhs, batch_size)
}

pub(crate) fn unify_bool<B: Backend>(
    lhs: BatchedPrimitive<BoolTensor<B>>,
    rhs: BatchedPrimitive<BoolTensor<B>>,
) -> (BoolTensor<B>, BoolTensor<B>, Option<usize>) {
    let (lhs, rhs, batch_size) = match (lhs.batch_size, rhs.batch_size) {
        (None, None) => return (lhs.primitive, rhs.primitive, None),
        (Some(n), None) => (lhs.primitive, promote_bool::<B>(rhs, n), Some(n)),
        (None, Some(n)) => (promote_bool::<B>(lhs, n), rhs.primitive, Some(n)),
        (Some(a), Some(b)) => {
            assert_eq!(a, b, "Vmap batch sizes must agree");
            (lhs.primitive, rhs.primitive, Some(a))
        }
    };
    let (lhs, rhs) = align_ranks_post_axis_0(lhs, rhs, |t, s| B::bool_reshape(t, s), |t| t.shape());
    (lhs, rhs, batch_size)
}

pub(crate) fn batched_axis(batch_size: Option<usize>, axis: usize) -> usize {
    match batch_size {
        Some(_) => axis + 1,
        None => axis,
    }
}

pub(crate) fn batched_shape(batch_size: Option<usize>, shape: &Shape) -> Shape {
    match batch_size {
        Some(n) => {
            let mut dims = Vec::with_capacity(shape.num_dims() + 1);
            dims.push(n);
            dims.extend_from_slice(shape.as_slice());
            Shape::from(dims)
        }
        None => shape.clone(),
    }
}

pub(crate) fn prepend_full_batch_slice(batch_size: Option<usize>, slices: &[Slice]) -> Vec<Slice> {
    match batch_size {
        Some(n) => {
            let mut out = Vec::with_capacity(slices.len() + 1);
            out.push(Slice::new(0, Some(n as isize), 1));
            out.extend_from_slice(slices);
            out
        }
        None => slices.to_vec(),
    }
}

/// Axis configuration for one leaf when using [`vmap_with_axes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VmapAxis {
    /// The leaf is batched along the given axis.
    Batched(usize),
    /// The leaf is not batched.
    Unbatched,
}

/// Vectorizes a function over physical axis 0 of every input leaf.
///
/// The input tree leaves are batched tensors (the batch dimension is part of their physical
/// shape). The function receives leaves whose logical shape excludes that dimension. Output
/// leaves whose physical batch dimension is absent are expanded to the active batch size before
/// being returned, so constant outputs and gradients that do not touch a batched activation are
/// still correctly vectorized.
pub fn vmap<B, F, Out, P>(f: F) -> impl Fn(&P) -> Out
where
    B: Backend,
    P: PyTree<B, WithBackend<B> = P>,
    Out: PyTree<B, WithBackend<B> = Out>,
    F: Fn(&P::WithBackend<Vmap<B>>) -> Out::WithBackend<Vmap<B>>,
{
    move |inputs| vmap_impl::<B, F, Out, P, _>(&f, inputs, |_| VmapAxis::Batched(0))
}

/// Vectorizes a function with an explicit axis configuration per input leaf.
///
/// The axes are consumed in deterministic leaf-visitation order.
pub fn vmap_with_axes<B, F, Out, P>(f: F, axes: Vec<VmapAxis>) -> impl Fn(&P) -> Out
where
    B: Backend,
    P: PyTree<B, WithBackend<B> = P>,
    Out: PyTree<B, WithBackend<B> = Out>,
    F: Fn(&P::WithBackend<Vmap<B>>) -> Out::WithBackend<Vmap<B>>,
{
    move |inputs| {
        assert_eq!(
            axes.len(),
            inputs.num_leaves(),
            "vmap_with_axes: number of axes ({}) does not match number of input leaves ({})",
            axes.len(),
            inputs.num_leaves(),
        );
        vmap_impl::<B, F, Out, P, _>(&f, inputs, |index| axes[index])
    }
}

fn vmap_impl<B, F, Out, P, A>(f: &F, inputs: &P, mut axis_of: A) -> Out
where
    B: Backend,
    P: PyTree<B, WithBackend<B> = P>,
    Out: PyTree<B, WithBackend<B> = Out>,
    F: Fn(&P::WithBackend<Vmap<B>>) -> Out::WithBackend<Vmap<B>>,
    A: FnMut(usize) -> VmapAxis,
{
    let mut index = 0;
    let batched = inputs.map_float::<Vmap<B>, _>(&mut |primitive: &FloatTensor<B>| {
        let axis = axis_of(index);
        index += 1;
        match axis {
            VmapAxis::Unbatched => BatchedPrimitive::unbatched(primitive.clone()),
            VmapAxis::Batched(axis) => {
                let primitive = if axis == 0 {
                    primitive.clone()
                } else {
                    B::float_swap_dims(primitive.clone(), axis, 0)
                };
                let batch_size = primitive.shape()[0];
                BatchedPrimitive::batched(primitive, batch_size)
            }
        }
    });

    let mut batch_size = None;
    batched.visit_float(&mut |leaf: &BatchedPrimitive<FloatTensor<B>>| {
        if let Some(size) = leaf.batch_size {
            batch_size.get_or_insert(size);
        }
    });
    let batch_size = batch_size.expect("vmap requires at least one batched input leaf");

    let output = f(&batched);

    output.map_float::<B, _>(&mut |leaf: &BatchedPrimitive<FloatTensor<B>>| {
        match leaf.batch_size {
            Some(size) => {
                assert_eq!(
                    size, batch_size,
                    "Vmap output batch size must match the input batch size"
                );
                leaf.primitive.clone()
            }
            None => {
                // The user-requested edge case: an unbatched output leaf (for example the
                // gradient of a linear operation) must still be replicated along the active
                // batch axis.
                let shape = leaf.primitive.shape();
                let logical = shape.as_slice();
                let mut reshape = Vec::with_capacity(logical.len() + 1);
                reshape.push(1);
                reshape.extend_from_slice(logical);
                let reshaped = B::float_reshape(leaf.primitive.clone(), Shape::from(reshape));
                let mut target = Vec::with_capacity(logical.len() + 1);
                target.push(batch_size);
                target.extend_from_slice(logical);
                B::float_expand(reshaped, Shape::from(target))
            }
        }
    })
}

impl<B: Backend> FloatTensorOps<Vmap<B>> for Vmap<B> {
    fn float_from_data(data: TensorData, device: &B::Device) -> Batched<FloatTensor<B>> {
        BatchedPrimitive::unbatched(B::float_from_data(data, device))
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        device: &B::Device,
        dtype: FloatDType,
    ) -> Batched<FloatTensor<B>> {
        BatchedPrimitive::unbatched(B::float_random(shape, distribution, device, dtype))
    }

    fn float_into_data(
        tensor: Batched<FloatTensor<B>>,
    ) -> impl core::future::Future<Output = Result<TensorData, ExecutionError>> + Send {
        B::float_into_data(tensor.primitive)
    }

    fn float_to_device(
        tensor: Batched<FloatTensor<B>>,
        device: &B::Device,
    ) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_to_device(tensor.primitive, device),
            batch_size: tensor.batch_size,
        }
    }

    fn float_into_int(
        tensor: Batched<FloatTensor<B>>,
        out_dtype: IntDType,
    ) -> Batched<IntTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_into_int(tensor.primitive, out_dtype),
            batch_size: tensor.batch_size,
        }
    }

    fn float_empty(shape: Shape, device: &B::Device, dtype: FloatDType) -> Batched<FloatTensor<B>> {
        BatchedPrimitive::unbatched(B::float_empty(shape, device, dtype))
    }

    fn float_add(
        lhs: Batched<FloatTensor<B>>,
        rhs: Batched<FloatTensor<B>>,
    ) -> Batched<FloatTensor<B>> {
        let (lhs, rhs, batch_size) = unify_float::<B>(lhs, rhs);
        BatchedPrimitive {
            primitive: B::float_add(lhs, rhs),
            batch_size,
        }
    }

    fn float_add_scalar(lhs: Batched<FloatTensor<B>>, rhs: Scalar) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_add_scalar(lhs.primitive, rhs),
            batch_size: lhs.batch_size,
        }
    }

    fn float_sub(
        lhs: Batched<FloatTensor<B>>,
        rhs: Batched<FloatTensor<B>>,
    ) -> Batched<FloatTensor<B>> {
        let (lhs, rhs, batch_size) = unify_float::<B>(lhs, rhs);
        BatchedPrimitive {
            primitive: B::float_sub(lhs, rhs),
            batch_size,
        }
    }

    fn float_sub_scalar(lhs: Batched<FloatTensor<B>>, rhs: Scalar) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_sub_scalar(lhs.primitive, rhs),
            batch_size: lhs.batch_size,
        }
    }

    fn float_mul(
        lhs: Batched<FloatTensor<B>>,
        rhs: Batched<FloatTensor<B>>,
    ) -> Batched<FloatTensor<B>> {
        let (lhs, rhs, batch_size) = unify_float::<B>(lhs, rhs);
        BatchedPrimitive {
            primitive: B::float_mul(lhs, rhs),
            batch_size,
        }
    }

    fn float_mul_scalar(lhs: Batched<FloatTensor<B>>, rhs: Scalar) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_mul_scalar(lhs.primitive, rhs),
            batch_size: lhs.batch_size,
        }
    }

    fn float_div(
        lhs: Batched<FloatTensor<B>>,
        rhs: Batched<FloatTensor<B>>,
    ) -> Batched<FloatTensor<B>> {
        let (lhs, rhs, batch_size) = unify_float::<B>(lhs, rhs);
        BatchedPrimitive {
            primitive: B::float_div(lhs, rhs),
            batch_size,
        }
    }

    fn float_div_scalar(lhs: Batched<FloatTensor<B>>, rhs: Scalar) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_div_scalar(lhs.primitive, rhs),
            batch_size: lhs.batch_size,
        }
    }

    fn float_remainder(
        lhs: Batched<FloatTensor<B>>,
        rhs: Batched<FloatTensor<B>>,
    ) -> Batched<FloatTensor<B>> {
        let (lhs, rhs, batch_size) = unify_float::<B>(lhs, rhs);
        BatchedPrimitive {
            primitive: B::float_remainder(lhs, rhs),
            batch_size,
        }
    }

    fn float_remainder_scalar(
        lhs: Batched<FloatTensor<B>>,
        rhs: Scalar,
    ) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_remainder_scalar(lhs.primitive, rhs),
            batch_size: lhs.batch_size,
        }
    }

    fn float_fmod(
        lhs: Batched<FloatTensor<B>>,
        rhs: Batched<FloatTensor<B>>,
    ) -> Batched<FloatTensor<B>> {
        let (lhs, rhs, batch_size) = unify_float::<B>(lhs, rhs);
        BatchedPrimitive {
            primitive: B::float_fmod(lhs, rhs),
            batch_size,
        }
    }

    fn float_fmod_scalar(lhs: Batched<FloatTensor<B>>, rhs: Scalar) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_fmod_scalar(lhs.primitive, rhs),
            batch_size: lhs.batch_size,
        }
    }

    fn float_matmul(
        lhs: Batched<FloatTensor<B>>,
        rhs: Batched<FloatTensor<B>>,
    ) -> Batched<FloatTensor<B>> {
        let (mut lhs_prim, mut rhs_prim, batch_size) = match (lhs.batch_size, rhs.batch_size) {
            (None, None) => {
                return BatchedPrimitive::unbatched(B::float_matmul(lhs.primitive, rhs.primitive));
            }
            (Some(n), None) => (lhs.primitive, promote_float::<B>(rhs, n), Some(n)),
            (None, Some(n)) => (promote_float::<B>(lhs, n), rhs.primitive, Some(n)),
            (Some(a), Some(b)) => {
                assert_eq!(a, b, "Vmap batch sizes must agree");
                (lhs.primitive, rhs.primitive, Some(a))
            }
        };

        let mut lhs_rank = lhs_prim.shape().num_dims();
        let mut rhs_rank = rhs_prim.shape().num_dims();

        let mut squeeze_lhs = false;
        let mut squeeze_rhs = false;

        if lhs_rank == 2 && rhs_rank >= 3 {
            // [n, K] -> [n, 1, K]
            let shape = lhs_prim.shape();
            let n = shape[0];
            let k = shape[1];
            lhs_prim = B::float_reshape(lhs_prim, Shape::new([n, 1, k]));
            lhs_rank = 3;
            squeeze_lhs = true;
        } else if rhs_rank == 2 && lhs_rank >= 3 {
            // [n, K] -> [n, K, 1]
            let shape = rhs_prim.shape();
            let n = shape[0];
            let k = shape[1];
            rhs_prim = B::float_reshape(rhs_prim, Shape::new([n, k, 1]));
            rhs_rank = 3;
            squeeze_rhs = true;
        } else if lhs_rank == 2 && rhs_rank == 2 {
            let shape_l = lhs_prim.shape();
            let shape_r = rhs_prim.shape();
            lhs_prim = B::float_reshape(lhs_prim, Shape::new([shape_l[0], 1, shape_l[1]]));
            rhs_prim = B::float_reshape(rhs_prim, Shape::new([shape_r[0], shape_r[1], 1]));
            lhs_rank = 3;
            rhs_rank = 3;
            squeeze_lhs = true;
            squeeze_rhs = true;
        }

        if lhs_rank < rhs_rank {
            let diff = rhs_rank - lhs_rank;
            let mut dims = lhs_prim.shape().as_slice().to_vec();
            for _ in 0..diff {
                dims.insert(1, 1);
            }
            lhs_prim = B::float_reshape(lhs_prim, Shape::from(dims));
        } else if rhs_rank < lhs_rank {
            let diff = lhs_rank - rhs_rank;
            let mut dims = rhs_prim.shape().as_slice().to_vec();
            for _ in 0..diff {
                dims.insert(1, 1);
            }
            rhs_prim = B::float_reshape(rhs_prim, Shape::from(dims));
        }

        let mut result = B::float_matmul(lhs_prim, rhs_prim);

        if squeeze_lhs || squeeze_rhs {
            let res_shape = result.shape();
            let dims = res_shape.as_slice();
            let r = dims.len();
            let mut new_dims = Vec::with_capacity(r);
            for (i, &d) in dims.iter().enumerate() {
                if squeeze_lhs && i == r - 2 {
                    continue;
                }
                if squeeze_rhs && i == r - 1 {
                    continue;
                }
                new_dims.push(d);
            }
            result = B::float_reshape(result, Shape::from(new_dims));
        }

        BatchedPrimitive {
            primitive: result,
            batch_size,
        }
    }

    fn float_cross(
        lhs: Batched<FloatTensor<B>>,
        rhs: Batched<FloatTensor<B>>,
        dim: usize,
    ) -> Batched<FloatTensor<B>> {
        let (lhs, rhs, batch_size) = unify_float::<B>(lhs, rhs);
        BatchedPrimitive {
            primitive: B::float_cross(lhs, rhs, batched_axis(batch_size, dim)),
            batch_size,
        }
    }

    fn float_recip(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_recip(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_swap_dims(
        tensor: Batched<FloatTensor<B>>,
        dim1: usize,
        dim2: usize,
    ) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_swap_dims(
                tensor.primitive,
                batched_axis(tensor.batch_size, dim1),
                batched_axis(tensor.batch_size, dim2),
            ),
            batch_size: tensor.batch_size,
        }
    }

    fn float_permute(tensor: Batched<FloatTensor<B>>, axes: &[usize]) -> Batched<FloatTensor<B>> {
        let physical = match tensor.batch_size {
            Some(_) => {
                let mut axes = axes.iter().map(|axis| axis + 1).collect::<Vec<_>>();
                axes.insert(0, 0);
                axes
            }
            None => axes.to_vec(),
        };
        BatchedPrimitive {
            primitive: B::float_permute(tensor.primitive, &physical),
            batch_size: tensor.batch_size,
        }
    }

    fn float_flip(tensor: Batched<FloatTensor<B>>, axes: &[usize]) -> Batched<FloatTensor<B>> {
        let axes = match tensor.batch_size {
            Some(_) => axes.iter().map(|axis| axis + 1).collect::<Vec<_>>(),
            None => axes.to_vec(),
        };
        BatchedPrimitive {
            primitive: B::float_flip(tensor.primitive, &axes),
            batch_size: tensor.batch_size,
        }
    }

    fn float_reshape(tensor: Batched<FloatTensor<B>>, shape: Shape) -> Batched<FloatTensor<B>> {
        let shape = batched_shape(tensor.batch_size, &shape);
        BatchedPrimitive {
            primitive: B::float_reshape(tensor.primitive, shape),
            batch_size: tensor.batch_size,
        }
    }

    fn float_gather(
        dim: usize,
        tensor: Batched<FloatTensor<B>>,
        indices: Batched<IntTensor<B>>,
    ) -> Batched<FloatTensor<B>> {
        let (tensor, indices, batch_size) = match (tensor.batch_size, indices.batch_size) {
            (None, None) => (tensor.primitive, indices.primitive, None),
            (Some(n), None) => (tensor.primitive, promote_int::<B>(indices, n), Some(n)),
            (None, Some(n)) => (promote_float::<B>(tensor, n), indices.primitive, Some(n)),
            (Some(a), Some(b)) => {
                assert_eq!(a, b);
                (tensor.primitive, indices.primitive, Some(a))
            }
        };
        BatchedPrimitive {
            primitive: B::float_gather(batched_axis(batch_size, dim), tensor, indices),
            batch_size,
        }
    }

    fn float_scatter(
        dim: usize,
        tensor: Batched<FloatTensor<B>>,
        indices: Batched<IntTensor<B>>,
        value: Batched<FloatTensor<B>>,
        update: burn_backend::tensor::IndexingUpdateOp,
    ) -> Batched<FloatTensor<B>> {
        let batch_size = tensor
            .batch_size
            .or(indices.batch_size)
            .or(value.batch_size);
        match batch_size {
            Some(n) => {
                let tensor = promote_float::<B>(tensor, n);
                let indices = promote_int::<B>(indices, n);
                let value = promote_float::<B>(value, n);
                BatchedPrimitive {
                    primitive: B::float_scatter(dim + 1, tensor, indices, value, update),
                    batch_size: Some(n),
                }
            }
            None => BatchedPrimitive::unbatched(B::float_scatter(
                dim,
                tensor.primitive,
                indices.primitive,
                value.primitive,
                update,
            )),
        }
    }

    fn float_select(
        tensor: Batched<FloatTensor<B>>,
        dim: usize,
        indices: Batched<IntTensor<B>>,
    ) -> Batched<FloatTensor<B>> {
        match (tensor.batch_size, indices.batch_size) {
            (None, None) => BatchedPrimitive::unbatched(B::float_select(
                tensor.primitive,
                dim,
                indices.primitive,
            )),
            (Some(n), None) => BatchedPrimitive::batched(
                B::float_select(tensor.primitive, dim + 1, indices.primitive),
                n,
            ),
            (None, Some(n)) => BatchedPrimitive::batched(
                B::float_gather(dim + 1, promote_float::<B>(tensor, n), indices.primitive),
                n,
            ),
            (Some(a), Some(b)) => {
                assert_eq!(a, b);
                BatchedPrimitive::batched(
                    B::float_gather(dim + 1, tensor.primitive, indices.primitive),
                    a,
                )
            }
        }
    }

    fn float_select_assign(
        tensor: Batched<FloatTensor<B>>,
        dim: usize,
        indices: Batched<IntTensor<B>>,
        value: Batched<FloatTensor<B>>,
        update: burn_backend::tensor::IndexingUpdateOp,
    ) -> Batched<FloatTensor<B>> {
        let batch_size = tensor
            .batch_size
            .or(indices.batch_size)
            .or(value.batch_size);
        match batch_size {
            Some(n) => {
                let tensor = promote_float::<B>(tensor, n);
                let indices = promote_int::<B>(indices, n);
                let value = promote_float::<B>(value, n);
                BatchedPrimitive::batched(
                    B::float_select_assign(tensor, dim + 1, indices, value, update),
                    n,
                )
            }
            None => BatchedPrimitive::unbatched(B::float_select_assign(
                tensor.primitive,
                dim,
                indices.primitive,
                value.primitive,
                update,
            )),
        }
    }

    fn float_slice(tensor: Batched<FloatTensor<B>>, slices: &[Slice]) -> Batched<FloatTensor<B>> {
        let slices = prepend_full_batch_slice(tensor.batch_size, slices);
        BatchedPrimitive {
            primitive: B::float_slice(tensor.primitive, &slices),
            batch_size: tensor.batch_size,
        }
    }

    fn float_slice_assign(
        tensor: Batched<FloatTensor<B>>,
        slices: &[Slice],
        value: Batched<FloatTensor<B>>,
    ) -> Batched<FloatTensor<B>> {
        let batch_size = tensor.batch_size.or(value.batch_size);
        let (tensor, value) = match batch_size {
            Some(n) => (promote_float::<B>(tensor, n), promote_float::<B>(value, n)),
            None => (tensor.primitive, value.primitive),
        };
        let slices = prepend_full_batch_slice(batch_size, slices);
        BatchedPrimitive {
            primitive: B::float_slice_assign(tensor, &slices, value),
            batch_size,
        }
    }

    fn float_mask_where(
        tensor: Batched<FloatTensor<B>>,
        mask: Batched<BoolTensor<B>>,
        value: Batched<FloatTensor<B>>,
    ) -> Batched<FloatTensor<B>> {
        let batch_size = tensor.batch_size.or(mask.batch_size).or(value.batch_size);
        match batch_size {
            Some(n) => {
                let tensor = promote_float::<B>(tensor, n);
                let mask = promote_bool::<B>(mask, n);
                let value = promote_float::<B>(value, n);
                BatchedPrimitive::batched(B::float_mask_where(tensor, mask, value), n)
            }
            None => BatchedPrimitive::unbatched(B::float_mask_where(
                tensor.primitive,
                mask.primitive,
                value.primitive,
            )),
        }
    }

    fn float_mask_fill(
        tensor: Batched<FloatTensor<B>>,
        mask: Batched<BoolTensor<B>>,
        value: Scalar,
    ) -> Batched<FloatTensor<B>> {
        let batch_size = tensor.batch_size.or(mask.batch_size);
        match batch_size {
            Some(n) => {
                let tensor = promote_float::<B>(tensor, n);
                let mask = promote_bool::<B>(mask, n);
                BatchedPrimitive::batched(B::float_mask_fill(tensor, mask, value), n)
            }
            None => BatchedPrimitive::unbatched(B::float_mask_fill(
                tensor.primitive,
                mask.primitive,
                value,
            )),
        }
    }

    async fn float_mask_select(
        tensor: Batched<FloatTensor<B>>,
        mask: Batched<BoolTensor<B>>,
    ) -> Batched<FloatTensor<B>> {
        assert!(
            tensor.batch_size.is_none() && mask.batch_size.is_none(),
            "float_mask_select is not supported on batched (vmap) tensors; \
             close over masks outside the vmap scope or call float_mask_select per-sample"
        );
        BatchedPrimitive::unbatched(B::float_mask_select(tensor.primitive, mask.primitive).await)
    }

    fn float_sum(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        match tensor.batch_size {
            None => BatchedPrimitive::unbatched(B::float_sum(tensor.primitive)),
            Some(n) => {
                let numel = tensor.primitive.shape().num_elements() / n;
                let reshaped =
                    B::float_reshape(tensor.primitive, Shape::from(alloc::vec![n, numel]));
                BatchedPrimitive::batched(B::float_sum_dim(reshaped, 1), n)
            }
        }
    }

    fn float_prod(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        match tensor.batch_size {
            None => BatchedPrimitive::unbatched(B::float_prod(tensor.primitive)),
            Some(n) => {
                let numel = tensor.primitive.shape().num_elements() / n;
                let reshaped =
                    B::float_reshape(tensor.primitive, Shape::from(alloc::vec![n, numel]));
                BatchedPrimitive::batched(B::float_prod_dim(reshaped, 1), n)
            }
        }
    }

    fn float_prod_dim(tensor: Batched<FloatTensor<B>>, dim: usize) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_prod_dim(tensor.primitive, batched_axis(tensor.batch_size, dim)),
            batch_size: tensor.batch_size,
        }
    }

    fn float_sum_dim(tensor: Batched<FloatTensor<B>>, dim: usize) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_sum_dim(tensor.primitive, batched_axis(tensor.batch_size, dim)),
            batch_size: tensor.batch_size,
        }
    }

    fn float_mean_dim(tensor: Batched<FloatTensor<B>>, dim: usize) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_mean_dim(tensor.primitive, batched_axis(tensor.batch_size, dim)),
            batch_size: tensor.batch_size,
        }
    }

    fn float_cumsum(tensor: Batched<FloatTensor<B>>, dim: usize) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_cumsum(tensor.primitive, batched_axis(tensor.batch_size, dim)),
            batch_size: tensor.batch_size,
        }
    }

    fn float_cumprod(tensor: Batched<FloatTensor<B>>, dim: usize) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_cumprod(tensor.primitive, batched_axis(tensor.batch_size, dim)),
            batch_size: tensor.batch_size,
        }
    }

    fn float_cummin(tensor: Batched<FloatTensor<B>>, dim: usize) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_cummin(tensor.primitive, batched_axis(tensor.batch_size, dim)),
            batch_size: tensor.batch_size,
        }
    }

    fn float_cummax(tensor: Batched<FloatTensor<B>>, dim: usize) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_cummax(tensor.primitive, batched_axis(tensor.batch_size, dim)),
            batch_size: tensor.batch_size,
        }
    }

    fn float_cast(tensor: Batched<FloatTensor<B>>, dtype: FloatDType) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_cast(tensor.primitive, dtype),
            batch_size: tensor.batch_size,
        }
    }

    fn float_exp(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_exp(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_log(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_log(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_log1p(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_log1p(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_powf(
        lhs: Batched<FloatTensor<B>>,
        rhs: Batched<FloatTensor<B>>,
    ) -> Batched<FloatTensor<B>> {
        let (lhs, rhs, batch_size) = unify_float::<B>(lhs, rhs);
        BatchedPrimitive {
            primitive: B::float_powf(lhs, rhs),
            batch_size,
        }
    }

    fn float_powf_scalar_impl(
        tensor: Batched<FloatTensor<B>>,
        value: Scalar,
    ) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_powf_scalar_impl(tensor.primitive, value),
            batch_size: tensor.batch_size,
        }
    }

    fn float_sqrt(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_sqrt(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_abs(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_abs(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_cos(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_cos(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_sin(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_sin(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_tan(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_tan(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_cosh(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_cosh(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_sinh(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_sinh(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_tanh(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_tanh(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_acos(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_acos(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_acosh(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_acosh(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_asin(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_asin(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_asinh(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_asinh(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_atan(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_atan(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_atanh(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_atanh(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_atan2(
        lhs: Batched<FloatTensor<B>>,
        rhs: Batched<FloatTensor<B>>,
    ) -> Batched<FloatTensor<B>> {
        let (lhs, rhs, batch_size) = unify_float::<B>(lhs, rhs);
        BatchedPrimitive {
            primitive: B::float_atan2(lhs, rhs),
            batch_size,
        }
    }

    fn float_round(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_round(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_floor(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_floor(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_ceil(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_ceil(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_trunc(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_trunc(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_erf(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_erf(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_clamp_min(tensor: Batched<FloatTensor<B>>, min: Scalar) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_clamp_min(tensor.primitive, min),
            batch_size: tensor.batch_size,
        }
    }

    fn float_clamp_max(tensor: Batched<FloatTensor<B>>, max: Scalar) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_clamp_max(tensor.primitive, max),
            batch_size: tensor.batch_size,
        }
    }

    fn float_clamp(
        tensor: Batched<FloatTensor<B>>,
        min: Scalar,
        max: Scalar,
    ) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_clamp(tensor.primitive, min, max),
            batch_size: tensor.batch_size,
        }
    }

    fn float_argmax(
        tensor: Batched<FloatTensor<B>>,
        dim: usize,
        out_dtype: IntDType,
    ) -> Batched<IntTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_argmax(
                tensor.primitive,
                batched_axis(tensor.batch_size, dim),
                out_dtype,
            ),
            batch_size: tensor.batch_size,
        }
    }

    fn float_argmin(
        tensor: Batched<FloatTensor<B>>,
        dim: usize,
        out_dtype: IntDType,
    ) -> Batched<IntTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_argmin(
                tensor.primitive,
                batched_axis(tensor.batch_size, dim),
                out_dtype,
            ),
            batch_size: tensor.batch_size,
        }
    }

    fn float_max_dim(tensor: Batched<FloatTensor<B>>, dim: usize) -> Batched<FloatTensor<B>> {
        let physical = batched_axis(tensor.batch_size, dim);
        BatchedPrimitive {
            primitive: B::float_max_dim(tensor.primitive, physical),
            batch_size: tensor.batch_size,
        }
    }

    fn float_max_dim_with_indices(
        tensor: Batched<FloatTensor<B>>,
        dim: usize,
        indices_dtype: IntDType,
    ) -> (Batched<FloatTensor<B>>, Batched<IntTensor<B>>) {
        let physical = batched_axis(tensor.batch_size, dim);
        let (values, indices) =
            B::float_max_dim_with_indices(tensor.primitive, physical, indices_dtype);
        (
            BatchedPrimitive {
                primitive: values,
                batch_size: tensor.batch_size,
            },
            BatchedPrimitive {
                primitive: indices,
                batch_size: tensor.batch_size,
            },
        )
    }

    fn float_min_dim(tensor: Batched<FloatTensor<B>>, dim: usize) -> Batched<FloatTensor<B>> {
        let physical = batched_axis(tensor.batch_size, dim);
        BatchedPrimitive {
            primitive: B::float_min_dim(tensor.primitive, physical),
            batch_size: tensor.batch_size,
        }
    }

    fn float_min_dim_with_indices(
        tensor: Batched<FloatTensor<B>>,
        dim: usize,
        indices_dtype: IntDType,
    ) -> (Batched<FloatTensor<B>>, Batched<IntTensor<B>>) {
        let physical = batched_axis(tensor.batch_size, dim);
        let (values, indices) =
            B::float_min_dim_with_indices(tensor.primitive, physical, indices_dtype);
        (
            BatchedPrimitive {
                primitive: values,
                batch_size: tensor.batch_size,
            },
            BatchedPrimitive {
                primitive: indices,
                batch_size: tensor.batch_size,
            },
        )
    }

    fn float_sort(
        tensor: Batched<FloatTensor<B>>,
        dim: usize,
        descending: bool,
    ) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_sort(
                tensor.primitive,
                batched_axis(tensor.batch_size, dim),
                descending,
            ),
            batch_size: tensor.batch_size,
        }
    }

    fn float_sort_with_indices(
        tensor: Batched<FloatTensor<B>>,
        dim: usize,
        descending: bool,
        indices_dtype: IntDType,
    ) -> (Batched<FloatTensor<B>>, Batched<IntTensor<B>>) {
        let physical = batched_axis(tensor.batch_size, dim);
        let (values, indices) =
            B::float_sort_with_indices(tensor.primitive, physical, descending, indices_dtype);
        (
            BatchedPrimitive {
                primitive: values,
                batch_size: tensor.batch_size,
            },
            BatchedPrimitive {
                primitive: indices,
                batch_size: tensor.batch_size,
            },
        )
    }

    fn float_argsort(
        tensor: Batched<FloatTensor<B>>,
        dim: usize,
        descending: bool,
        out_dtype: IntDType,
    ) -> Batched<IntTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_argsort(
                tensor.primitive,
                batched_axis(tensor.batch_size, dim),
                descending,
                out_dtype,
            ),
            batch_size: tensor.batch_size,
        }
    }

    fn float_sign(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_sign(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_expand(tensor: Batched<FloatTensor<B>>, shape: Shape) -> Batched<FloatTensor<B>> {
        let shape = batched_shape(tensor.batch_size, &shape);
        BatchedPrimitive {
            primitive: B::float_expand(tensor.primitive, shape),
            batch_size: tensor.batch_size,
        }
    }

    fn float_unfold(
        tensor: Batched<FloatTensor<B>>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_unfold(
                tensor.primitive,
                batched_axis(tensor.batch_size, dim),
                size,
                step,
            ),
            batch_size: tensor.batch_size,
        }
    }

    fn float_cat(tensors: Vec<Batched<FloatTensor<B>>>, dim: usize) -> Batched<FloatTensor<B>> {
        let batch_size = tensors.iter().find_map(|tensor| tensor.batch_size);
        match batch_size {
            None => BatchedPrimitive::unbatched(B::float_cat(
                tensors.into_iter().map(|tensor| tensor.primitive).collect(),
                dim,
            )),
            Some(n) => {
                let tensors = tensors
                    .into_iter()
                    .map(|tensor| promote_float::<B>(tensor, n))
                    .collect();
                BatchedPrimitive::batched(B::float_cat(tensors, dim + 1), n)
            }
        }
    }

    fn float_pad(
        tensor: Batched<FloatTensor<B>>,
        padding: &[(usize, usize)],
        mode: PadMode,
    ) -> Batched<FloatTensor<B>> {
        let padding = match tensor.batch_size {
            Some(_) => {
                let mut out = alloc::vec![(0, 0)];
                out.extend_from_slice(padding);
                out
            }
            None => padding.to_vec(),
        };
        BatchedPrimitive {
            primitive: B::float_pad(tensor.primitive, &padding, mode),
            batch_size: tensor.batch_size,
        }
    }

    fn float_repeat_dim(
        tensor: Batched<FloatTensor<B>>,
        dim: usize,
        times: usize,
    ) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_repeat_dim(
                tensor.primitive,
                batched_axis(tensor.batch_size, dim),
                times,
            ),
            batch_size: tensor.batch_size,
        }
    }

    fn float_detach(tensor: Batched<FloatTensor<B>>) -> Batched<FloatTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_detach(tensor.primitive),
            batch_size: tensor.batch_size,
        }
    }

    fn float_set_require_grad(
        tensor: Batched<FloatTensor<B>>,
        _require_grad: bool,
    ) -> Batched<FloatTensor<B>> {
        tensor
    }

    fn float_is_require_grad(_tensor: &Batched<FloatTensor<B>>) -> bool {
        false
    }

    fn float_equal(
        lhs: Batched<FloatTensor<B>>,
        rhs: Batched<FloatTensor<B>>,
        out_dtype: BoolDType,
    ) -> Batched<BoolTensor<B>> {
        let (lhs, rhs, batch_size) = unify_float::<B>(lhs, rhs);
        BatchedPrimitive {
            primitive: B::float_equal(lhs, rhs, out_dtype),
            batch_size,
        }
    }

    fn float_not_equal(
        lhs: Batched<FloatTensor<B>>,
        rhs: Batched<FloatTensor<B>>,
        out_dtype: BoolDType,
    ) -> Batched<BoolTensor<B>> {
        let (lhs, rhs, batch_size) = unify_float::<B>(lhs, rhs);
        BatchedPrimitive {
            primitive: B::float_not_equal(lhs, rhs, out_dtype),
            batch_size,
        }
    }

    fn float_equal_elem(
        lhs: Batched<FloatTensor<B>>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> Batched<BoolTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_equal_elem(lhs.primitive, rhs, out_dtype),
            batch_size: lhs.batch_size,
        }
    }

    fn float_not_equal_elem(
        lhs: Batched<FloatTensor<B>>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> Batched<BoolTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_not_equal_elem(lhs.primitive, rhs, out_dtype),
            batch_size: lhs.batch_size,
        }
    }

    fn float_greater(
        lhs: Batched<FloatTensor<B>>,
        rhs: Batched<FloatTensor<B>>,
        out_dtype: BoolDType,
    ) -> Batched<BoolTensor<B>> {
        let (lhs, rhs, batch_size) = unify_float::<B>(lhs, rhs);
        BatchedPrimitive {
            primitive: B::float_greater(lhs, rhs, out_dtype),
            batch_size,
        }
    }

    fn float_greater_elem(
        lhs: Batched<FloatTensor<B>>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> Batched<BoolTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_greater_elem(lhs.primitive, rhs, out_dtype),
            batch_size: lhs.batch_size,
        }
    }

    fn float_greater_equal(
        lhs: Batched<FloatTensor<B>>,
        rhs: Batched<FloatTensor<B>>,
        out_dtype: BoolDType,
    ) -> Batched<BoolTensor<B>> {
        let (lhs, rhs, batch_size) = unify_float::<B>(lhs, rhs);
        BatchedPrimitive {
            primitive: B::float_greater_equal(lhs, rhs, out_dtype),
            batch_size,
        }
    }

    fn float_greater_equal_elem(
        lhs: Batched<FloatTensor<B>>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> Batched<BoolTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_greater_equal_elem(lhs.primitive, rhs, out_dtype),
            batch_size: lhs.batch_size,
        }
    }

    fn float_lower(
        lhs: Batched<FloatTensor<B>>,
        rhs: Batched<FloatTensor<B>>,
        out_dtype: BoolDType,
    ) -> Batched<BoolTensor<B>> {
        let (lhs, rhs, batch_size) = unify_float::<B>(lhs, rhs);
        BatchedPrimitive {
            primitive: B::float_lower(lhs, rhs, out_dtype),
            batch_size,
        }
    }

    fn float_lower_elem(
        lhs: Batched<FloatTensor<B>>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> Batched<BoolTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_lower_elem(lhs.primitive, rhs, out_dtype),
            batch_size: lhs.batch_size,
        }
    }

    fn float_lower_equal(
        lhs: Batched<FloatTensor<B>>,
        rhs: Batched<FloatTensor<B>>,
        out_dtype: BoolDType,
    ) -> Batched<BoolTensor<B>> {
        let (lhs, rhs, batch_size) = unify_float::<B>(lhs, rhs);
        BatchedPrimitive {
            primitive: B::float_lower_equal(lhs, rhs, out_dtype),
            batch_size,
        }
    }

    fn float_lower_equal_elem(
        lhs: Batched<FloatTensor<B>>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> Batched<BoolTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_lower_equal_elem(lhs.primitive, rhs, out_dtype),
            batch_size: lhs.batch_size,
        }
    }

    fn float_any(tensor: Batched<FloatTensor<B>>, out_dtype: BoolDType) -> Batched<BoolTensor<B>> {
        match tensor.batch_size {
            None => BatchedPrimitive::unbatched(B::float_any(tensor.primitive, out_dtype)),
            Some(n) => {
                let numel = tensor.primitive.shape().num_elements() / n;
                let reshaped =
                    B::float_reshape(tensor.primitive, Shape::from(alloc::vec![n, numel]));
                BatchedPrimitive::batched(B::float_any_dim(reshaped, 1, out_dtype), n)
            }
        }
    }

    fn float_any_dim(
        tensor: Batched<FloatTensor<B>>,
        dim: usize,
        out_dtype: BoolDType,
    ) -> Batched<BoolTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_any_dim(
                tensor.primitive,
                batched_axis(tensor.batch_size, dim),
                out_dtype,
            ),
            batch_size: tensor.batch_size,
        }
    }

    fn float_all(tensor: Batched<FloatTensor<B>>, out_dtype: BoolDType) -> Batched<BoolTensor<B>> {
        match tensor.batch_size {
            None => BatchedPrimitive::unbatched(B::float_all(tensor.primitive, out_dtype)),
            Some(n) => {
                let numel = tensor.primitive.shape().num_elements() / n;
                let reshaped =
                    B::float_reshape(tensor.primitive, Shape::from(alloc::vec![n, numel]));
                BatchedPrimitive::batched(B::float_all_dim(reshaped, 1, out_dtype), n)
            }
        }
    }

    fn float_all_dim(
        tensor: Batched<FloatTensor<B>>,
        dim: usize,
        out_dtype: BoolDType,
    ) -> Batched<BoolTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_all_dim(
                tensor.primitive,
                batched_axis(tensor.batch_size, dim),
                out_dtype,
            ),
            batch_size: tensor.batch_size,
        }
    }

    fn float_is_nan(
        tensor: Batched<FloatTensor<B>>,
        out_dtype: BoolDType,
    ) -> Batched<BoolTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_is_nan(tensor.primitive, out_dtype),
            batch_size: tensor.batch_size,
        }
    }

    fn float_is_inf(
        tensor: Batched<FloatTensor<B>>,
        out_dtype: BoolDType,
    ) -> Batched<BoolTensor<B>> {
        BatchedPrimitive {
            primitive: B::float_is_inf(tensor.primitive, out_dtype),
            batch_size: tensor.batch_size,
        }
    }

    fn float_grid_sample_2d(
        tensor: Batched<FloatTensor<B>>,
        grid: Batched<FloatTensor<B>>,
        options: burn_backend::ops::GridSampleOptions,
    ) -> Batched<FloatTensor<B>> {
        let (tensor, grid, batch_size) = unify_float::<B>(tensor, grid);
        BatchedPrimitive {
            primitive: B::float_grid_sample_2d(tensor, grid, options),
            batch_size,
        }
    }
}
