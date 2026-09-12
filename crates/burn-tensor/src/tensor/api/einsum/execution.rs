//! Execute the shared equation plan using existing differentiable operations.
//!
//! The literal macro emits the same operation chain directly. Only dimension
//! arithmetic and shape-dependent broadcasting decisions remain on the host.

use super::EinsumOperand;
use crate::{DType, Int, Tensor, kind::Numeric, ops::BridgeTensor};
use alloc::{vec, vec::Vec};
use burn_einsum::{Axis, ContractionPlan, InputPlan, Plan};
use core::marker::PhantomData;

/// Rank-erased tensor operations used by the generated lowering.
pub struct Value<K: Numeric> {
    primitive: BridgeTensor,
    kind: PhantomData<K>,
}

impl<K: Numeric> Value<K> {
    fn new(primitive: BridgeTensor) -> Self {
        Self {
            primitive,
            kind: PhantomData,
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        self.primitive.shape().to_vec()
    }

    pub fn diagonal(
        self,
        first: usize,
        second: usize,
        permutation: &[usize],
        restore: &[usize],
    ) -> Self {
        Self::new(diagonal::<K>(
            self.primitive,
            first,
            second,
            permutation,
            restore,
        ))
    }

    pub fn permute(self, axes: &[usize]) -> Self {
        Self::new(permute::<K>(self.primitive, axes))
    }

    pub fn reshape(self, shape: Vec<usize>) -> Self {
        if &self.primitive.shape()[..] == shape.as_slice() {
            self
        } else {
            Self::new(K::reshape(self.primitive, shape.into()))
        }
    }

    pub fn sum_dims(mut self, axes: &[usize]) -> Self {
        for &axis in axes {
            if self.primitive.shape()[axis] != 1 {
                self.primitive = K::sum_dim(self.primitive, axis);
            }
        }
        self
    }

    pub fn mul(self, other: Self) -> Self {
        Self::new(K::mul(self.primitive, other.primitive))
    }

    pub fn matmul(self, other: Self) -> Self {
        Self::new(K::matmul(self.primitive, other.primitive))
    }

    pub fn finish<const D: usize>(self, output_rank: usize) -> Tensor<D, K> {
        check_output_rank::<D>(output_rank);
        let shape = if output_rank == 0 {
            vec![1]
        } else {
            self.primitive.shape()[..output_rank].to_vec()
        };
        Tensor::new(self.reshape(shape).primitive)
    }
}

/// Validated inputs and the widths needed to bind symbolic ellipsis axes.
pub struct Prepared<K: Numeric> {
    pub operands: alloc::vec::IntoIter<Value<K>>,
    pub ellipsis_width: usize,
    pub output_rank: usize,
}

/// Validate input metadata and bind the plan to operand ranks.
pub fn prepare<K: Numeric>(
    operands: impl IntoIterator<Item = EinsumOperand<K>>,
    ranks: &[(usize, bool)],
    output_dimensions: usize,
    total_dimensions: usize,
    ellipsis: Option<usize>,
) -> Prepared<K> {
    let operands: Vec<_> = operands.into_iter().map(|op| op.primitive).collect();
    assert!(
        !operands.is_empty(),
        "einsum: at least one operand is required"
    );
    assert_eq!(
        operands.len(),
        ranks.len(),
        "einsum: operand count does not match the equation"
    );
    let device = K::device(&operands[0]);
    let dtype = operands[0].dtype();
    assert!(
        !matches!(dtype, DType::QFloat(_)),
        "einsum: quantized operands are not supported"
    );
    let mut ellipsis_width = 0;
    for (index, (operand, &(named_rank, has_ellipsis))) in operands.iter().zip(ranks).enumerate() {
        assert_eq!(
            K::device(operand),
            device,
            "einsum: operands must be on the same device"
        );
        assert_eq!(
            operand.dtype(),
            dtype,
            "einsum: operands must have the same dtype"
        );
        let shape = operand.shape();
        if has_ellipsis {
            assert!(
                shape.len() >= named_rank,
                "einsum: operand {index} has fewer dimensions than subscripts"
            );
            ellipsis_width = ellipsis_width.max(shape.len() - named_rank);
        } else if named_rank == 0 {
            assert_eq!(
                &shape[..],
                &[1],
                "einsum: scalar operand {index} must have shape [1]"
            );
        } else {
            assert_eq!(
                shape.len(),
                named_rank,
                "einsum: rank of operand {index} does not match its subscripts"
            );
        }
    }
    debug_assert!(output_dimensions <= total_dimensions);
    let output_rank = if ellipsis.is_some_and(|axis| axis < output_dimensions) {
        output_dimensions - 1 + ellipsis_width
    } else {
        output_dimensions
    };
    Prepared {
        operands: operands
            .into_iter()
            .map(Value::new)
            .collect::<Vec<_>>()
            .into_iter(),
        ellipsis_width,
        output_rank,
    }
}

fn check_output_rank<const D: usize>(output_rank: usize) {
    assert_eq!(
        D,
        output_rank.max(1),
        "einsum: output rank does not match the equation (scalar results have rank 1)"
    );
}

/// Bind a planned axis sequence to the actual ellipsis width.
pub fn axes(planned: &[Axis], width: usize) -> Vec<usize> {
    let mut axes = Vec::new();
    for axis in planned {
        match *axis {
            Axis::Index(index) => axes.push(index),
            Axis::AfterEllipsis(index) => axes.push(index + width),
            Axis::Ellipsis(start) => axes.extend(start..start + width),
        }
    }
    axes
}

fn axis(axis: Axis, width: usize) -> usize {
    match axis {
        Axis::Index(index) => index,
        Axis::AfterEllipsis(index) => index + width,
        Axis::Ellipsis(_) => unreachable!("a diagonal always references a named axis"),
    }
}

/// Insert the planned singleton axes after permuting the axes present in an input.
pub fn alignment_shape(
    shape: &[usize],
    present: &[bool],
    ellipsis: Option<usize>,
    global_width: usize,
) -> Vec<usize> {
    let named = present
        .iter()
        .enumerate()
        .filter(|&(index, &exists)| exists && Some(index) != ellipsis)
        .count();
    let local_width = if ellipsis.is_some_and(|index| present[index]) {
        shape.len() - named
    } else {
        0
    };
    let mut source = 0;
    let mut aligned = Vec::new();
    for (index, &exists) in present.iter().enumerate() {
        if Some(index) == ellipsis {
            aligned.extend(core::iter::repeat_n(1, global_width - local_width));
            aligned.extend_from_slice(&shape[source..source + local_width]);
            source += local_width;
        } else if exists {
            aligned.push(shape[source]);
            source += 1;
        } else {
            aligned.push(1);
        }
    }
    if aligned.is_empty() {
        aligned.push(1);
    }
    aligned
}

/// Check aligned input sizes and record the last non-singleton use of each axis.
pub fn validate_broadcast<K: Numeric>(operands: &[&Value<K>]) -> Vec<usize> {
    let mut sizes = vec![1; operands[0].primitive.shape().len()];
    let mut last_use = vec![0; sizes.len()];
    for (index, operand) in operands.iter().enumerate() {
        for (axis, (size, dimension)) in sizes
            .iter_mut()
            .zip(operand.primitive.shape().iter())
            .enumerate()
        {
            if *dimension != 1 {
                assert!(
                    *size == 1 || *size == *dimension,
                    "einsum: operand {index} has incompatible broadcast dimensions"
                );
                *size = *dimension;
                last_use[axis] = index;
            }
        }
    }
    last_use
}

/// Whether the equation's matrix grouping applies without broadcast adaptation.
pub fn can_matmul(
    left: &[usize],
    right: &[usize],
    shared: &[usize],
    left_axes: &[usize],
    right_axes: &[usize],
    contraction: &[usize],
) -> bool {
    !left.contains(&0)
        && !right.contains(&0)
        && shared
            .iter()
            .chain(contraction)
            .all(|&axis| left[axis] == right[axis])
        && left_axes.iter().all(|&axis| right[axis] == 1)
        && right_axes.iter().all(|&axis| left[axis] == 1)
}

/// Concrete reshape sizes for the already selected matrix grouping.
pub struct MatmulShapes {
    pub left: Vec<usize>,
    pub right: Vec<usize>,
    pub output: Vec<usize>,
}

/// Evaluate reshape dimensions for the planned matrix groups.
pub fn matmul_shapes(
    left: &[usize],
    right: &[usize],
    shared: &[usize],
    left_axes: &[usize],
    right_axes: &[usize],
    contraction: &[usize],
) -> MatmulShapes {
    let batch = dimension_product(left, shared);
    let rows = dimension_product(left, left_axes);
    let columns = dimension_product(right, right_axes);
    let contracted = dimension_product(left, contraction);
    let mut output: Vec<_> = shared
        .iter()
        .chain(left_axes)
        .map(|&axis| left[axis])
        .collect();
    output.extend(core::iter::repeat_n(1, contraction.len()));
    output.extend(right_axes.iter().map(|&axis| right[axis]));
    MatmulShapes {
        left: vec![batch, rows, contracted],
        right: vec![batch, contracted, columns],
        output,
    }
}

/// Adapt a planned contraction when singleton or empty dimensions change its lowering.
pub fn broadcast_contract<K: Numeric>(
    mut left: Value<K>,
    mut right: Value<K>,
    contraction: &[usize],
) -> Value<K> {
    let left_shape = left.shape();
    let right_shape = right.shape();
    let mut sum = Vec::new();
    for &dimension in contraction {
        match (left_shape[dimension] != 1, right_shape[dimension] != 1) {
            (true, true) => sum.push(dimension),
            (true, false) => left = left.sum_dims(&[dimension]),
            (false, true) => right = right.sum_dims(&[dimension]),
            (false, false) => {}
        }
    }
    if sum.is_empty() {
        left.mul(right)
    } else {
        Value::new(sumproduct_pair::<K>(left.primitive, right.primitive, &sum))
    }
}

/// Whether broadcasting lets a later-use axis be contracted at this step.
pub fn can_contract_early(
    left: &[usize],
    right: &[usize],
    deferred: &[usize],
    last_use: &[usize],
    operand_index: usize,
) -> bool {
    deferred
        .iter()
        .any(|&axis| last_use[axis] <= operand_index && (left[axis] != 1 || right[axis] != 1))
}

/// Contract axes whose remaining equation occurrences all have size one.
pub fn contract_early<K: Numeric>(
    left: Value<K>,
    right: Value<K>,
    contraction: &[usize],
    deferred: &[usize],
    last_use: &[usize],
    operand_index: usize,
) -> Value<K> {
    let mut contraction = contraction.to_vec();
    contraction.extend(
        deferred
            .iter()
            .copied()
            .filter(|&axis| last_use[axis] <= operand_index),
    );
    contraction.sort_unstable();
    broadcast_contract(left, right, &contraction)
}

fn align<K: Numeric>(
    mut value: Value<K>,
    input: &InputPlan,
    plan: &Plan,
    width: usize,
) -> Value<K> {
    let local_width = if input.has_ellipsis {
        value.shape().len() - input.named_rank
    } else {
        0
    };
    for diagonal in &input.diagonals {
        value = value.diagonal(
            axis(diagonal.first, local_width),
            axis(diagonal.second, local_width),
            &axes(&diagonal.permutation, local_width),
            &axes(&diagonal.restore, local_width),
        );
    }
    let value = value.permute(&axes(&input.permutation, local_width));
    let shape = alignment_shape(&value.shape(), &input.axes, plan.ellipsis, width);
    value.reshape(shape)
}

fn contract<K: Numeric>(
    left: Value<K>,
    right: Value<K>,
    plan: &ContractionPlan,
    width: usize,
    last_use: &[usize],
    operand_index: usize,
) -> Value<K> {
    let mut left = left.sum_dims(&axes(&plan.left_reduce, width));
    let mut right = right.sum_dims(&axes(&plan.right_reduce, width));
    let deferred = axes(&plan.deferred_reduce, width);
    if !deferred.is_empty()
        && can_contract_early(
            &left.shape(),
            &right.shape(),
            &deferred,
            last_use,
            operand_index,
        )
    {
        let contraction = plan
            .matmul
            .as_ref()
            .map(|matrix| axes(&matrix.contraction, width))
            .unwrap_or_default();
        return contract_early(
            left,
            right,
            &contraction,
            &deferred,
            last_use,
            operand_index,
        );
    }
    let Some(matrix) = &plan.matmul else {
        return left.mul(right);
    };
    if matrix.swap {
        core::mem::swap(&mut left, &mut right);
    }
    let shared = axes(&matrix.shared, width);
    let left_axes = axes(&matrix.left, width);
    let right_axes = axes(&matrix.right, width);
    let contraction = axes(&matrix.contraction, width);
    let left_shape = left.shape();
    let right_shape = right.shape();
    if can_matmul(
        &left_shape,
        &right_shape,
        &shared,
        &left_axes,
        &right_axes,
        &contraction,
    ) {
        let shapes = matmul_shapes(
            &left_shape,
            &right_shape,
            &shared,
            &left_axes,
            &right_axes,
            &contraction,
        );
        left.permute(&axes(&matrix.left_permutation, width))
            .reshape(shapes.left)
            .matmul(
                right
                    .permute(&axes(&matrix.right_permutation, width))
                    .reshape(shapes.right),
            )
            .reshape(shapes.output)
            .permute(&axes(&matrix.output_permutation, width))
    } else {
        broadcast_contract(left, right, &contraction)
    }
}

pub(super) fn execute<const D: usize, K: Numeric>(
    plan: &Plan,
    operands: impl IntoIterator<Item = EinsumOperand<K>>,
) -> Tensor<D, K> {
    let ranks: Vec<_> = plan
        .inputs
        .iter()
        .map(|input| (input.named_rank, input.has_ellipsis))
        .collect();
    let prepared = prepare(
        operands,
        &ranks,
        plan.output_dimensions,
        plan.total_dimensions,
        plan.ellipsis,
    );
    check_output_rank::<D>(prepared.output_rank);
    let aligned: Vec<_> = prepared
        .operands
        .zip(&plan.inputs)
        .map(|(value, input)| align(value, input, plan, prepared.ellipsis_width))
        .collect();
    let last_use = validate_broadcast(&aligned.iter().collect::<Vec<_>>());
    let mut operands = aligned.into_iter();
    let mut result = operands.next().unwrap();
    for (index, (right, contraction)) in operands.zip(&plan.contractions).enumerate() {
        result = contract(
            result,
            right,
            contraction,
            prepared.ellipsis_width,
            &last_use,
            index + 1,
        );
    }
    result
        .sum_dims(&axes(&plan.final_reduce, prepared.ellipsis_width))
        .finish(prepared.output_rank)
}

fn diagonal<K: Numeric>(
    tensor: BridgeTensor,
    first: usize,
    second: usize,
    permutation: &[usize],
    restore: &[usize],
) -> BridgeTensor {
    let shape = tensor.shape();
    let size = shape[first];
    assert_eq!(
        size, shape[second],
        "einsum: repeated subscripts must have equal dimensions"
    );
    let remaining = &permutation[..permutation.len() - 2];
    let mut flattened: Vec<_> = remaining.iter().map(|&dim| shape[dim]).collect();
    flattened.push(
        size.checked_mul(size)
            .expect("einsum: diagonal size overflow"),
    );
    let device = K::device(&tensor);
    let tensor = K::reshape(permute::<K>(tensor, permutation), flattened.into());
    // As in linalg::diag, select diagonal positions instead of multiplying by
    // an identity mask: off-diagonal NaNs must not contaminate the result.
    let end = i64::try_from(size).expect("einsum: diagonal index overflow");
    let step = end.checked_add(1).expect("einsum: diagonal index overflow");
    // Index width must not inherit a configurable I8/I16 default: even a
    // small matrix can have diagonal offsets outside that range.
    let index_dtype = if size.saturating_mul(size) <= i32::MAX as usize {
        DType::I32
    } else {
        DType::I64
    };
    let indices = Tensor::<1, Int>::arange(0..end, (&device, index_dtype)) * step;
    let tensor = K::select(tensor, remaining.len(), indices.primitive);
    permute::<K>(tensor, restore)
}

fn sumproduct_pair<K: Numeric>(
    mut left: BridgeTensor,
    mut right: BridgeTensor,
    sum_dimensions: &[usize],
) -> BridgeTensor {
    let left_shape = left.shape();
    let right_shape = right.shape();
    if left_shape.contains(&0) || right_shape.contains(&0) {
        // Some backends cannot execute an empty matmul. This product is also
        // empty, so the equivalent multiply/reduce path needs no large
        // intermediate and retains both operands in the autodiff graph.
        let mut result = K::mul(left, right);
        for &dimension in sum_dimensions {
            result = K::sum_dim(result, dimension);
        }
        return result;
    }
    let mut shared = Vec::new();
    let mut left_output = Vec::new();
    let mut right_output = Vec::new();
    for dim in 0..left_shape.len() {
        if sum_dimensions.contains(&dim) {
            continue;
        }
        match (left_shape[dim] != 1, right_shape[dim] != 1) {
            (true, true) => shared.push(dim),
            (true, false) => left_output.push(dim),
            _ => right_output.push(dim),
        }
    }
    // Preserve the reference's orientation optimization for output axes that
    // place the right operand entirely before the left operand.
    if let (Some(&last_right), Some(&first_left)) = (right_output.last(), left_output.first())
        && last_right < first_left
    {
        core::mem::swap(&mut left, &mut right);
        core::mem::swap(&mut left_output, &mut right_output);
    }
    let left_shape = left.shape();
    let right_shape = right.shape();
    let batch = dimension_product(&left_shape, &shared);
    let rows = dimension_product(&left_shape, &left_output);
    let columns = dimension_product(&right_shape, &right_output);
    let contraction = dimension_product(&left_shape, sum_dimensions);
    let mut output_shape: Vec<_> = shared
        .iter()
        .chain(&left_output)
        .map(|&d| left_shape[d])
        .collect();
    output_shape.extend(core::iter::repeat_n(1, sum_dimensions.len()));
    output_shape.extend(right_output.iter().map(|&d| right_shape[d]));
    let left_permutation: Vec<_> = shared
        .iter()
        .chain(&left_output)
        .chain(sum_dimensions)
        .chain(&right_output)
        .copied()
        .collect();
    let right_permutation: Vec<_> = shared
        .iter()
        .chain(sum_dimensions)
        .chain(&right_output)
        .chain(&left_output)
        .copied()
        .collect();
    let mut output_permutation = vec![0; left_permutation.len()];
    for (grouped, &original) in left_permutation.iter().enumerate() {
        output_permutation[original] = grouped;
    }
    let left = K::reshape(
        permute::<K>(left, &left_permutation),
        [batch, rows, contraction].into(),
    );
    let right = K::reshape(
        permute::<K>(right, &right_permutation),
        [batch, contraction, columns].into(),
    );
    let result = K::matmul(left, right);
    let result = K::reshape(result, output_shape.into());
    permute::<K>(result, &output_permutation)
}

fn dimension_product(shape: &[usize], axes: &[usize]) -> usize {
    axes.iter()
        .try_fold(1usize, |product, &axis| product.checked_mul(shape[axis]))
        .expect("einsum: dimension product overflow")
}

fn permute<K: Numeric>(tensor: BridgeTensor, axes: &[usize]) -> BridgeTensor {
    if axes.iter().enumerate().all(|(index, &axis)| index == axis) {
        tensor
    } else {
        K::permute(tensor, axes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trailing_singleton_weights_allow_contraction_before_large_broadcast_product() {
        // ij,jk,j->ik: waiting for the last scalar weight would materialize
        // [1024, 1024, 4096], although a matrix product needs only [1024, 1024].
        // Exercise the branch decision using metadata without that allocation.
        let left = [1024, 1, 4096];
        let right = [1, 1024, 4096];
        assert!(can_contract_early(&left, &right, &[2], &[0, 1, 1], 1));
        assert!(!can_contract_early(&left, &right, &[2], &[0, 1, 2], 1));
        assert!(!can_contract_early(
            &[1024, 1, 1],
            &[1, 1024, 1],
            &[2],
            &[0, 1, 1],
            1
        ));
    }
}
