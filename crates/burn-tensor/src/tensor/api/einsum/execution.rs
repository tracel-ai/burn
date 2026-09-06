//! Alignment and left-to-right sumproduct algorithm adapted from
//! <https://github.com/Mikyx-1/pytorch-einsum-reference/tree/0451b648390965f341f5f9f8fe69c2f64a5259a5>.
//! Shape bookkeeping stays on the host; tensor values remain on their device.

use alloc::{collections::VecDeque, vec, vec::Vec};
use burn_einsum::{ELLIPSIS, Equation};

use crate::{DType, Int, Tensor, kind::Numeric, ops::BridgeTensor};

pub(super) struct Layout {
    pub output_dimensions: usize,
    total_dimensions: usize,
    named_dimensions: [usize; 52],
    ellipsis_start: usize,
    ellipsis_dimensions: usize,
    input_ellipsis_dimensions: Vec<usize>,
}

impl Layout {
    pub fn new<K: Numeric>(equation: &Equation, operands: &[BridgeTensor]) -> Self {
        assert!(
            !operands.is_empty(),
            "einsum: at least one operand is required"
        );
        assert_eq!(
            equation.inputs.len(),
            operands.len(),
            "einsum: operand count does not match the equation"
        );
        let device = K::device(&operands[0]);
        let dtype = operands[0].dtype();
        assert!(
            !matches!(dtype, DType::QFloat(_)),
            "einsum: quantized operands are not supported"
        );
        let mut input_ellipsis_dimensions = Vec::with_capacity(operands.len());
        let mut present = [false; 52];
        for (index, (labels, operand)) in equation.inputs.iter().zip(operands).enumerate() {
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
            let ellipses = labels.iter().filter(|&&label| label == ELLIPSIS).count();
            assert!(ellipses <= 1, "einsum: repeated input ellipsis");
            for &label in labels {
                if label != ELLIPSIS {
                    assert!(label < 52, "einsum: invalid label");
                    present[label as usize] = true;
                }
            }
            let named = labels.len() - ellipses;
            let ellipsis_width = if labels.is_empty() {
                assert_eq!(
                    &shape[..],
                    &[1],
                    "einsum: scalar operand {index} must have shape [1]"
                );
                0
            } else if ellipses == 1 {
                assert!(
                    shape.len() >= named,
                    "einsum: operand {index} has fewer dimensions than subscripts"
                );
                shape.len() - named
            } else {
                assert_eq!(
                    shape.len(),
                    named,
                    "einsum: rank of operand {index} does not match its subscripts"
                );
                0
            };
            input_ellipsis_dimensions.push(ellipsis_width);
        }
        let ellipsis_dimensions = input_ellipsis_dimensions.iter().copied().max().unwrap_or(0);
        let mut named_dimensions = [usize::MAX; 52];
        let mut dimension = 0;
        let mut ellipsis_start = None;
        for &label in &equation.output {
            if label == ELLIPSIS {
                assert!(ellipsis_start.is_none(), "einsum: repeated output ellipsis");
                ellipsis_start = Some(dimension);
                dimension += ellipsis_dimensions;
            } else {
                assert!(
                    label < 52 && present[label as usize],
                    "einsum: output label is absent from inputs"
                );
                assert_eq!(
                    named_dimensions[label as usize],
                    usize::MAX,
                    "einsum: repeated output label"
                );
                named_dimensions[label as usize] = dimension;
                dimension += 1;
            }
        }
        let output_dimensions = dimension;
        let ellipsis_start = ellipsis_start.unwrap_or_else(|| {
            let start = dimension;
            dimension += ellipsis_dimensions;
            start
        });
        for label in 0..52 {
            if present[label] && named_dimensions[label] == usize::MAX {
                named_dimensions[label] = dimension;
                dimension += 1;
            }
        }
        Self {
            output_dimensions,
            // A scalar-only expression still uses Burn's physical [1] shape.
            total_dimensions: dimension.max(1),
            named_dimensions,
            ellipsis_start,
            ellipsis_dimensions,
            input_ellipsis_dimensions,
        }
    }

    fn expanded_labels(&self, labels: &[u8], operand: usize) -> Vec<usize> {
        let mut expanded = Vec::new();
        for &label in labels {
            if label == ELLIPSIS {
                let first = self.ellipsis_dimensions - self.input_ellipsis_dimensions[operand];
                expanded
                    .extend((first..self.ellipsis_dimensions).map(|dim| self.ellipsis_start + dim));
            } else {
                expanded.push(self.named_dimensions[label as usize]);
            }
        }
        expanded
    }
}

pub(super) fn align<K: Numeric>(
    equation: &Equation,
    layout: &Layout,
    operands: Vec<BridgeTensor>,
) -> (VecDeque<BridgeTensor>, Vec<usize>) {
    let mut sizes = vec![1; layout.total_dimensions];
    let mut counts = vec![0; layout.total_dimensions];
    let mut aligned = VecDeque::with_capacity(operands.len());
    for (index, (labels, mut operand)) in equation.inputs.iter().zip(operands).enumerate() {
        let mut axes = layout.expanded_labels(labels, index);
        let mut source = 0;
        while source < axes.len() {
            if let Some(previous) = axes[..source].iter().position(|&axis| axis == axes[source]) {
                operand = diagonal::<K>(operand, previous, source);
                axes.remove(source);
            } else {
                source += 1;
            }
        }
        let shape = operand.shape();
        let mut permutation = vec![usize::MAX; layout.total_dimensions];
        for (source, &target) in axes.iter().enumerate() {
            let size = shape[source];
            if size != 1 {
                assert!(
                    sizes[target] == 1 || sizes[target] == size,
                    "einsum: operand {index} has incompatible broadcast dimensions"
                );
                sizes[target] = size;
                counts[target] += 1;
            }
            permutation[target] = source;
        }
        let mut expanded_shape = if axes.is_empty() {
            Vec::new()
        } else {
            shape.to_vec()
        };
        for axis in &mut permutation {
            if *axis == usize::MAX {
                *axis = expanded_shape.len();
                expanded_shape.push(1);
            }
        }
        operand = K::reshape(operand, expanded_shape.into());
        aligned.push_back(permute::<K>(operand, &permutation));
    }
    (aligned, counts)
}

fn diagonal<K: Numeric>(tensor: BridgeTensor, first: usize, second: usize) -> BridgeTensor {
    let shape = tensor.shape();
    let size = shape[first];
    assert_eq!(
        size, shape[second],
        "einsum: repeated subscripts must have equal dimensions"
    );
    let remaining: Vec<_> = (0..shape.len())
        .filter(|&dim| dim != first && dim != second)
        .collect();
    let mut permutation = remaining.clone();
    permutation.extend([first, second]);
    let mut flattened: Vec<_> = remaining.iter().map(|&dim| shape[dim]).collect();
    flattened.push(
        size.checked_mul(size)
            .expect("einsum: diagonal size overflow"),
    );
    let device = K::device(&tensor);
    let tensor = K::reshape(permute::<K>(tensor, &permutation), flattened.into());
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
    let mut diagonal_order = remaining;
    diagonal_order.push(first);
    let restore: Vec<_> = (0..shape.len())
        .filter(|&dim| dim != second)
        .map(|dim| diagonal_order.iter().position(|&axis| axis == dim).unwrap())
        .collect();
    permute::<K>(tensor, &restore)
}

pub(super) fn contract_pair<K: Numeric>(
    mut left: BridgeTensor,
    mut right: BridgeTensor,
    layout: &Layout,
    counts: &mut [usize],
) -> BridgeTensor {
    let left_shape = left.shape();
    let right_shape = right.shape();
    let mut sum_dimensions = Vec::new();
    for dimension in layout.output_dimensions..layout.total_dimensions {
        let left_nontrivial = left_shape[dimension] != 1;
        let right_nontrivial = right_shape[dimension] != 1;
        if left_nontrivial && right_nontrivial {
            counts[dimension] -= 1;
            if counts[dimension] == 1 {
                sum_dimensions.push(dimension);
                counts[dimension] = 0;
            }
        } else if counts[dimension] == 1 {
            if left_nontrivial {
                left = K::sum_dim(left, dimension);
                counts[dimension] = 0;
            } else if right_nontrivial {
                right = K::sum_dim(right, dimension);
                counts[dimension] = 0;
            }
        }
    }
    if sum_dimensions.is_empty() {
        return K::mul(left, right);
    }
    sumproduct_pair::<K>(left, right, &sum_dimensions)
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

pub(super) fn finalize<K: Numeric>(mut result: BridgeTensor, layout: &Layout) -> BridgeTensor {
    for dim in layout.output_dimensions..layout.total_dimensions {
        if result.shape()[dim] != 1 {
            result = K::sum_dim(result, dim);
        }
    }
    let shape = result.shape();
    let output_shape = if layout.output_dimensions == 0 {
        vec![1]
    } else {
        shape[..layout.output_dimensions].to_vec()
    };
    K::reshape(result, output_shape.into())
}
