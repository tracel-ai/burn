//! Equation-dependent execution planning, with symbolic ellipsis dimensions.

use alloc::{vec, vec::Vec};

use crate::{ELLIPSIS, Equation};

/// An axis or contiguous ellipsis block in a tensor layout.
///
/// Resolve against the operand's ellipsis width for input alignment and
/// diagonals, or the maximum input ellipsis width for contractions. A width of
/// zero removes an [`Axis::Ellipsis`] block and leaves its neighboring axes
/// adjacent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// A concrete axis before an ellipsis, or in a layout without one.
    Index(usize),
    /// An axis whose index is this offset plus the ellipsis width.
    AfterEllipsis(usize),
    /// All axes from this start up to the start plus the ellipsis width.
    Ellipsis(usize),
}

/// Extract a repeated label's diagonal, preserving the first axis's position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagonal {
    /// The first occurrence in the current operand layout.
    pub first: Axis,
    /// The repeated occurrence removed from the current operand layout.
    pub second: Axis,
    /// Input axes ordered as remaining axes, first occurrence, second occurrence.
    pub permutation: Vec<Axis>,
    /// Restore the first occurrence's position after selecting the diagonal.
    ///
    /// Indices refer to the remaining axes followed by the selected diagonal axis.
    pub restore: Vec<Axis>,
}

/// Align an operand with the equation's canonical layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputPlan {
    /// Number of named label occurrences before extracting diagonals.
    pub named_rank: usize,
    /// Whether the operand subscript contains an ellipsis.
    pub has_ellipsis: bool,
    /// Diagonals in execution order, with indices updated after each extraction.
    pub diagonals: Vec<Diagonal>,
    /// Post-diagonal input axes in canonical order, omitting absent axes.
    pub permutation: Vec<Axis>,
    /// Presence of each canonical slot, treating the ellipsis as one slot.
    pub axes: Vec<bool>,
}

/// Lower a pair's shared contracted axes to a batched matrix multiplication.
///
/// Groups depend on label presence, so the executor must still handle
/// broadcasting, singleton contraction axes, and empty dimensions according to
/// the actual shapes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatmulPlan {
    /// Whether to exchange the left and right operands before applying the plan.
    pub swap: bool,
    /// Non-contracted axes present in both operands.
    pub shared: Vec<Axis>,
    /// Non-contracted axes present only in the left operand.
    pub left: Vec<Axis>,
    /// Remaining non-contracted axes, including absent singleton slots.
    pub right: Vec<Axis>,
    /// Axes present in both operands whose last use is this contraction.
    pub contraction: Vec<Axis>,
    /// Canonical axes ordered as shared, left, contraction, right.
    pub left_permutation: Vec<Axis>,
    /// Canonical axes ordered as shared, contraction, right, left.
    pub right_permutation: Vec<Axis>,
    /// Restore canonical order from the shared, left, contraction, right layout.
    ///
    /// Symbolic indices refer to the ellipsis position in that grouped layout.
    pub output_permutation: Vec<Axis>,
}

/// One left-to-right contraction with the next input operand.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContractionPlan {
    /// Omitted axes present in this pair but also used by later operands.
    ///
    /// Actual shapes may permit reducing these axes early when all later uses
    /// have singleton dimensions.
    pub deferred_reduce: Vec<Axis>,
    /// Last-use axes present only on the left, reduced before combining operands.
    pub left_reduce: Vec<Axis>,
    /// Last-use axes present only on the right, reduced before combining operands.
    pub right_reduce: Vec<Axis>,
    /// Matrix multiplication lowering, or elementwise multiplication if absent.
    pub matmul: Option<MatmulPlan>,
}

/// A shape-independent execution plan for an einsum equation.
///
/// Canonical slots contain output labels in their requested order, an omitted
/// input ellipsis if any, then remaining named labels in alphabetical order.
/// The ellipsis occupies one symbolic slot even when its runtime width is zero.
/// Scalar-only equations use one absent dummy slot for Burn's physical `[1]`
/// scalar representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Operand diagonal extraction and alignment plans.
    pub inputs: Vec<InputPlan>,
    /// Left-to-right combinations of the accumulator with each subsequent input.
    pub contractions: Vec<ContractionPlan>,
    /// The ellipsis's canonical slot, when any input contains an ellipsis.
    pub ellipsis: Option<usize>,
    /// Number of canonical output slots, counting an ellipsis as one slot.
    pub output_dimensions: usize,
    /// Number of canonical slots, including at least one physical scalar slot.
    pub total_dimensions: usize,
    /// Omitted axes still requiring reduction after the last contraction.
    pub final_reduce: Vec<Axis>,
}

impl Plan {
    pub(crate) fn new(equation: &Equation) -> Self {
        let has_ellipsis = equation
            .inputs
            .iter()
            .any(|input| input.contains(&ELLIPSIS));
        let mut canonical: Vec<_> = equation
            .output
            .iter()
            .copied()
            .filter(|&label| label != ELLIPSIS || has_ellipsis)
            .collect();
        let output_dimensions = canonical.len();
        if has_ellipsis && !canonical.contains(&ELLIPSIS) {
            canonical.push(ELLIPSIS);
        }
        let ellipsis = canonical.iter().position(|&label| label == ELLIPSIS);
        let mut present = [false; ELLIPSIS as usize];
        for input in &equation.inputs {
            for &label in input {
                if label != ELLIPSIS {
                    present[label as usize] = true;
                }
            }
        }
        for label in 0..ELLIPSIS {
            if present[label as usize] && !canonical.contains(&label) {
                canonical.push(label);
            }
        }
        if canonical.is_empty() {
            // This value cannot occur in a parsed equation.
            canonical.push(ELLIPSIS + 1);
        }
        let total_dimensions = canonical.len();
        let inputs: Vec<_> = equation
            .inputs
            .iter()
            .map(|input| InputPlan::new(input, &canonical))
            .collect();
        let mut last = vec![0; total_dimensions];
        for (index, input) in inputs.iter().enumerate() {
            for (slot, &present) in input.axes.iter().enumerate() {
                if present {
                    last[slot] = index;
                }
            }
        }
        let mut alive = inputs[0].axes.clone();
        let mut contractions = Vec::with_capacity(inputs.len() - 1);
        for (index, input) in inputs.iter().enumerate().skip(1) {
            let mut right = input.axes.clone();
            let mut deferred_reduce = Vec::new();
            let mut left_reduce = Vec::new();
            let mut right_reduce = Vec::new();
            let mut contracted = Vec::new();
            for slot in output_dimensions..total_dimensions {
                if last[slot] > index {
                    if alive[slot] || right[slot] {
                        deferred_reduce.push(axis(slot, ellipsis));
                    }
                    continue;
                }
                match (alive[slot], right[slot]) {
                    (true, true) => contracted.push(slot),
                    (true, false) => left_reduce.push(axis(slot, ellipsis)),
                    (false, true) => right_reduce.push(axis(slot, ellipsis)),
                    (false, false) => {}
                }
                alive[slot] = false;
                right[slot] = false;
            }
            let matmul = (!contracted.is_empty())
                .then(|| MatmulPlan::new(&alive, &right, &contracted, ellipsis));
            contractions.push(ContractionPlan {
                deferred_reduce,
                left_reduce,
                right_reduce,
                matmul,
            });
            for (left, right) in alive.iter_mut().zip(right) {
                *left |= right;
            }
        }
        let final_reduce = (output_dimensions..total_dimensions)
            .filter(|&slot| alive[slot])
            .map(|slot| axis(slot, ellipsis))
            .collect();
        Self {
            inputs,
            contractions,
            ellipsis,
            output_dimensions,
            total_dimensions,
            final_reduce,
        }
    }
}

impl InputPlan {
    fn new(input: &[u8], canonical: &[u8]) -> Self {
        let has_ellipsis = input.contains(&ELLIPSIS);
        let named_rank = input.len() - usize::from(has_ellipsis);
        let mut labels = input.to_vec();
        let mut diagonals = Vec::new();
        let mut source = 0;
        while source < labels.len() {
            if let Some(first) = labels[..source]
                .iter()
                .position(|&label| label == labels[source])
            {
                let ellipsis = labels.iter().position(|&label| label == ELLIPSIS);
                let mut diagonal_order: Vec<_> = (0..labels.len())
                    .filter(|&slot| slot != first && slot != source)
                    .collect();
                diagonal_order.push(first);
                let mut permutation: Vec<_> = diagonal_order
                    .iter()
                    .map(|&slot| axis(slot, ellipsis))
                    .collect();
                permutation.push(axis(source, ellipsis));
                let grouped_ellipsis = ellipsis
                    .and_then(|ellipsis| diagonal_order.iter().position(|&slot| slot == ellipsis));
                let restore = (0..labels.len())
                    .filter(|&slot| slot != source)
                    .map(|slot| {
                        let grouped = diagonal_order
                            .iter()
                            .position(|&axis| axis == slot)
                            .unwrap();
                        axis(grouped, grouped_ellipsis)
                    })
                    .collect();
                diagonals.push(Diagonal {
                    first: axis(first, ellipsis),
                    second: axis(source, ellipsis),
                    permutation,
                    restore,
                });
                labels.remove(source);
            } else {
                source += 1;
            }
        }
        let ellipsis = labels.iter().position(|&label| label == ELLIPSIS);
        let mut permutation = Vec::with_capacity(labels.len());
        let axes = canonical
            .iter()
            .map(|label| {
                if let Some(source) = labels.iter().position(|current| current == label) {
                    permutation.push(axis(source, ellipsis));
                    true
                } else {
                    false
                }
            })
            .collect();
        Self {
            named_rank,
            has_ellipsis,
            diagonals,
            permutation,
            axes,
        }
    }
}

impl MatmulPlan {
    fn new(
        left_axes: &[bool],
        right_axes: &[bool],
        contracted: &[usize],
        ellipsis: Option<usize>,
    ) -> Self {
        let mut shared = Vec::new();
        let mut left = Vec::new();
        let mut right = Vec::new();
        for slot in 0..left_axes.len() {
            if contracted.contains(&slot) {
                continue;
            }
            match (left_axes[slot], right_axes[slot]) {
                (true, true) => shared.push(slot),
                (true, false) => left.push(slot),
                _ => right.push(slot),
            }
        }
        // Keep the output's right-before-left orientation without consulting shapes.
        let swap =
            matches!((right.last(), left.first()), (Some(last), Some(first)) if last < first);
        if swap {
            core::mem::swap(&mut left, &mut right);
        }
        let left_order: Vec<_> = shared
            .iter()
            .chain(&left)
            .chain(contracted)
            .chain(&right)
            .copied()
            .collect();
        let right_order = shared
            .iter()
            .chain(contracted)
            .chain(&right)
            .chain(&left)
            .copied();
        let grouped_ellipsis =
            ellipsis.and_then(|ellipsis| left_order.iter().position(|&slot| slot == ellipsis));
        let mut output_permutation = vec![Axis::Index(0); left_order.len()];
        for (grouped, &original) in left_order.iter().enumerate() {
            output_permutation[original] = axis(grouped, grouped_ellipsis);
        }
        Self {
            swap,
            shared: shared.iter().map(|&slot| axis(slot, ellipsis)).collect(),
            left: left.iter().map(|&slot| axis(slot, ellipsis)).collect(),
            right: right.iter().map(|&slot| axis(slot, ellipsis)).collect(),
            contraction: contracted
                .iter()
                .map(|&slot| axis(slot, ellipsis))
                .collect(),
            left_permutation: left_order
                .iter()
                .map(|&slot| axis(slot, ellipsis))
                .collect(),
            right_permutation: right_order.map(|slot| axis(slot, ellipsis)).collect(),
            output_permutation,
        }
    }
}

fn axis(slot: usize, ellipsis: Option<usize>) -> Axis {
    match ellipsis {
        Some(start) if slot == start => Axis::Ellipsis(start),
        Some(start) if slot > start => Axis::AfterEllipsis(slot - 1),
        _ => Axis::Index(slot),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Axis::{AfterEllipsis, Ellipsis, Index};

    fn plan(equation: &str) -> Plan {
        Equation::parse(equation).unwrap().plan()
    }

    fn expand(axes: &[Axis], width: usize) -> Vec<usize> {
        axes.iter()
            .flat_map(|&axis| match axis {
                Index(index) => index..index + 1,
                AfterEllipsis(offset) => offset + width..offset + width + 1,
                Ellipsis(start) => start..start + width,
            })
            .collect()
    }

    #[test]
    fn matrix_multiplication_has_fixed_alignment_and_lowering() {
        let plan = plan("ij,jk->ik");
        assert_eq!(plan.ellipsis, None);
        assert_eq!(plan.output_dimensions, 2);
        assert_eq!(plan.total_dimensions, 3);
        assert_eq!(plan.inputs[0].permutation, vec![Index(0), Index(1)]);
        assert_eq!(plan.inputs[0].axes, vec![true, false, true]);
        assert_eq!(plan.inputs[1].permutation, vec![Index(1), Index(0)]);
        assert_eq!(plan.inputs[1].axes, vec![false, true, true]);
        assert_eq!(plan.contractions[0].left_reduce, vec![]);
        assert_eq!(plan.contractions[0].right_reduce, vec![]);
        assert_eq!(
            plan.contractions[0].matmul,
            Some(MatmulPlan {
                swap: false,
                shared: vec![],
                left: vec![Index(0)],
                right: vec![Index(1)],
                contraction: vec![Index(2)],
                left_permutation: vec![Index(0), Index(2), Index(1)],
                right_permutation: vec![Index(2), Index(1), Index(0)],
                output_permutation: vec![Index(0), Index(2), Index(1)],
            })
        );
        assert!(plan.final_reduce.is_empty());
    }

    #[test]
    fn chain_contracts_each_label_at_its_last_participating_operand() {
        let plan = plan("ij,jk,kl->il");
        assert_eq!(plan.contractions.len(), 2);
        assert_eq!(
            plan.contractions[0].matmul.as_ref().unwrap().contraction,
            vec![Index(2)]
        );
        assert_eq!(
            plan.contractions[1].matmul.as_ref().unwrap().contraction,
            vec![Index(3)]
        );
        // The retired j slot remains as a singleton in the canonical layout.
        assert_eq!(
            plan.contractions[1].matmul.as_ref().unwrap().right,
            vec![Index(1), Index(2)]
        );
        assert!(plan.final_reduce.is_empty());
    }

    #[test]
    fn labels_needed_by_later_operands_are_retained_in_the_accumulator() {
        let repeated = plan("i,i,i->");
        assert!(repeated.contractions[0].matmul.is_none());
        assert!(repeated.contractions[0].left_reduce.is_empty());
        assert!(repeated.contractions[0].right_reduce.is_empty());
        assert_eq!(
            repeated.contractions[1]
                .matmul
                .as_ref()
                .unwrap()
                .contraction,
            vec![Index(0)]
        );

        let skipped = plan("i,j,i->j");
        assert!(skipped.contractions[0].matmul.is_none());
        assert!(skipped.contractions[0].left_reduce.is_empty());
        assert_eq!(
            skipped.contractions[1].matmul.as_ref().unwrap().contraction,
            vec![Index(1)]
        );
    }

    #[test]
    fn later_uses_defer_contraction_but_allow_shape_dependent_early_reduction() {
        let plan = plan("ij,jk,j->ik");
        assert_eq!(plan.contractions[0].deferred_reduce, vec![Index(2)]);
        assert!(plan.contractions[0].matmul.is_none());
        assert!(plan.contractions[0].left_reduce.is_empty());
        assert!(plan.contractions[0].right_reduce.is_empty());
        assert!(plan.contractions[1].deferred_reduce.is_empty());
        assert_eq!(
            plan.contractions[1].matmul.as_ref().unwrap().contraction,
            vec![Index(2)]
        );
    }

    #[test]
    fn deferred_reductions_exclude_output_and_absent_axes() {
        let output = plan("i,i,i->i");
        assert!(
            output
                .contractions
                .iter()
                .all(|step| step.deferred_reduce.is_empty())
        );

        let absent = plan("i,j,k,k->i");
        assert!(absent.contractions[0].deferred_reduce.is_empty());
        assert_eq!(absent.contractions[1].deferred_reduce, vec![Index(2)]);
        assert!(absent.contractions[2].deferred_reduce.is_empty());
    }

    #[test]
    fn unilateral_last_use_axes_are_reduced_before_multiplication() {
        let plan = plan("ij,kl->ik");
        assert_eq!(plan.contractions[0].left_reduce, vec![Index(2)]);
        assert_eq!(plan.contractions[0].right_reduce, vec![Index(3)]);
        assert!(plan.contractions[0].matmul.is_none());
        assert!(plan.final_reduce.is_empty());
    }

    #[test]
    fn output_labels_are_never_retired() {
        let plan = plan("ij,ij,ij->ji");
        for input in &plan.inputs {
            assert_eq!(input.permutation, vec![Index(1), Index(0)]);
        }
        for contraction in &plan.contractions {
            assert!(contraction.left_reduce.is_empty());
            assert!(contraction.right_reduce.is_empty());
            assert!(contraction.matmul.is_none());
        }
        assert!(plan.final_reduce.is_empty());
    }

    #[test]
    fn single_operand_transposition_and_reduction_are_planned() {
        let plan = plan("ji->i");
        assert_eq!(plan.inputs[0].permutation, vec![Index(1), Index(0)]);
        assert!(plan.contractions.is_empty());
        assert_eq!(plan.final_reduce, vec![Index(1)]);
        assert_eq!(plan.output_dimensions, 1);
    }

    #[test]
    fn diagonals_update_indices_after_each_extraction() {
        let plan = plan("ijiji->ij");
        assert_eq!(plan.inputs[0].named_rank, 5);
        assert_eq!(
            plan.inputs[0]
                .diagonals
                .iter()
                .map(|diagonal| (diagonal.first, diagonal.second))
                .collect::<Vec<_>>(),
            vec![
                (Index(0), Index(2)),
                (Index(1), Index(2)),
                (Index(0), Index(2)),
            ]
        );
        assert_eq!(plan.inputs[0].permutation, vec![Index(0), Index(1)]);
    }

    #[test]
    fn diagonal_indices_follow_ellipsis_position_as_labels_are_removed() {
        let crossed = plan("i...ii->...");
        assert_eq!(crossed.inputs[0].named_rank, 3);
        assert_eq!(
            crossed.inputs[0]
                .diagonals
                .iter()
                .map(|diagonal| (diagonal.first, diagonal.second))
                .collect::<Vec<_>>(),
            vec![(Index(0), AfterEllipsis(1)), (Index(0), AfterEllipsis(1)),]
        );
        assert_eq!(crossed.inputs[0].permutation, vec![Ellipsis(1), Index(0)]);
        assert_eq!(crossed.final_reduce, vec![AfterEllipsis(0)]);

        let shifted = plan("iii...j->j...i");
        assert_eq!(shifted.inputs[0].named_rank, 4);
        assert_eq!(
            shifted.inputs[0]
                .diagonals
                .iter()
                .map(|diagonal| (diagonal.first, diagonal.second))
                .collect::<Vec<_>>(),
            vec![(Index(0), Index(1)), (Index(0), Index(1)),]
        );
        assert_eq!(
            shifted.inputs[0].permutation,
            vec![AfterEllipsis(1), Ellipsis(1), Index(0)]
        );

        let trailing = plan("...iii->...i");
        assert_eq!(
            trailing.inputs[0]
                .diagonals
                .iter()
                .map(|diagonal| (diagonal.first, diagonal.second))
                .collect::<Vec<_>>(),
            vec![
                (AfterEllipsis(0), AfterEllipsis(1)),
                (AfterEllipsis(0), AfterEllipsis(1)),
            ]
        );
    }

    #[test]
    fn output_ellipsis_without_input_ellipsis_has_no_slot() {
        assert_eq!(plan("ij,jk->i...k"), plan("ij,jk->ik"));
        assert_eq!(plan("ii"), plan("ii->"));
    }

    #[test]
    fn omitted_ellipsis_is_reduced_as_a_symbolic_block() {
        let single = plan("...i->i");
        assert_eq!(single.ellipsis, Some(1));
        assert_eq!(single.output_dimensions, 1);
        assert_eq!(single.final_reduce, vec![Ellipsis(1)]);
        assert_eq!(
            single.inputs[0].permutation,
            vec![AfterEllipsis(0), Ellipsis(0)]
        );

        let shared = plan("...i,...i->i");
        let matmul = shared.contractions[0].matmul.as_ref().unwrap();
        assert_eq!(matmul.shared, vec![Index(0)]);
        assert_eq!(matmul.contraction, vec![Ellipsis(1)]);
        assert!(shared.final_reduce.is_empty());

        let unilateral = plan("...i,i->i");
        assert_eq!(unilateral.contractions[0].left_reduce, vec![Ellipsis(1)]);
        assert!(unilateral.contractions[0].matmul.is_none());
    }

    #[test]
    fn symbolic_output_permutation_uses_the_grouped_ellipsis_position() {
        let plan = plan("...ij,...jk->i...k");
        let matmul = plan.contractions[0].matmul.as_ref().unwrap();
        assert_eq!(plan.ellipsis, Some(1));
        assert_eq!(matmul.shared, vec![Ellipsis(1)]);
        assert_eq!(matmul.left, vec![Index(0)]);
        assert_eq!(matmul.right, vec![AfterEllipsis(1)]);
        assert_eq!(matmul.contraction, vec![AfterEllipsis(2)]);
        assert_eq!(
            matmul.left_permutation,
            vec![Ellipsis(1), Index(0), AfterEllipsis(2), AfterEllipsis(1)]
        );
        assert_eq!(
            matmul.output_permutation,
            vec![
                AfterEllipsis(0),
                Ellipsis(0),
                AfterEllipsis(2),
                AfterEllipsis(1)
            ]
        );
    }

    #[test]
    fn grouped_permutations_are_invertible_for_zero_and_multiple_ellipsis_axes() {
        for equation in [
            "...ij,...jk->i...k",
            "i...j,...jk->ki...",
            "...ij,j...k->ki",
            "ij...,...jk->...ik",
            "...ij,jk,kl->li...",
        ] {
            let plan = plan(equation);
            for contraction in &plan.contractions {
                let matmul = contraction.matmul.as_ref().unwrap();
                for width in [0, 1, 3] {
                    let left = expand(&matmul.left_permutation, width);
                    let right = expand(&matmul.right_permutation, width);
                    let inverse = expand(&matmul.output_permutation, width);
                    let expected: Vec<_> = (0..plan.total_dimensions - 1 + width).collect();
                    assert_eq!(
                        inverse.iter().map(|&axis| left[axis]).collect::<Vec<_>>(),
                        expected
                    );
                    let mut right = right;
                    right.sort_unstable();
                    assert_eq!(right, expected);
                }
            }
        }
    }

    #[test]
    fn reversed_output_order_swaps_matmul_operands() {
        let plan = plan("ij,jk->ki");
        let matmul = plan.contractions[0].matmul.as_ref().unwrap();
        assert!(matmul.swap);
        assert_eq!(matmul.left, vec![Index(0)]);
        assert_eq!(matmul.right, vec![Index(1)]);
        assert_eq!(matmul.contraction, vec![Index(2)]);
        assert_eq!(matmul.left_permutation, vec![Index(0), Index(2), Index(1)]);
        assert_eq!(matmul.right_permutation, vec![Index(2), Index(1), Index(0)]);
        assert_eq!(
            matmul.output_permutation,
            vec![Index(0), Index(2), Index(1)]
        );
    }

    #[test]
    fn crossed_ellipsis_diagonal_moves_the_block_and_restores_it() {
        let plan = plan("i...i->...");
        let diagonal = &plan.inputs[0].diagonals[0];
        assert_eq!(diagonal.first, Index(0));
        assert_eq!(diagonal.second, AfterEllipsis(1));
        assert_eq!(
            diagonal.permutation,
            vec![Ellipsis(1), Index(0), AfterEllipsis(1)]
        );
        assert_eq!(diagonal.restore, vec![AfterEllipsis(0), Ellipsis(0)]);
    }

    #[test]
    fn diagonal_permutations_restore_axis_order_for_all_ellipsis_widths() {
        for equation in [
            "i...i->...",
            "...iii->...i",
            "iii...j->j...i",
            "i...jiji->ji...",
        ] {
            let plan = plan(equation);
            let input = &plan.inputs[0];
            for width in [0, 1, 3] {
                for (index, diagonal) in input.diagonals.iter().enumerate() {
                    let rank = input.named_rank + width - index;
                    let second = expand(&[diagonal.second], width)[0];
                    let mut selected = expand(&diagonal.permutation, width);
                    assert_eq!(selected.pop(), Some(second));
                    let restore = expand(&diagonal.restore, width);
                    let expected: Vec<_> = (0..rank).filter(|&axis| axis != second).collect();
                    assert_eq!(
                        restore
                            .iter()
                            .map(|&axis| selected[axis])
                            .collect::<Vec<_>>(),
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn scalars_preserve_an_absent_physical_slot() {
        let scalar = plan("");
        assert_eq!(scalar.output_dimensions, 0);
        assert_eq!(scalar.total_dimensions, 1);
        assert_eq!(scalar.ellipsis, None);
        assert_eq!(scalar.inputs[0].named_rank, 0);
        assert!(!scalar.inputs[0].has_ellipsis);
        assert!(scalar.inputs[0].permutation.is_empty());
        assert_eq!(scalar.inputs[0].axes, vec![false]);
        assert!(scalar.final_reduce.is_empty());

        let product = plan(",->...");
        assert_eq!(product.output_dimensions, 0);
        assert_eq!(product.total_dimensions, 1);
        assert!(product.contractions[0].matmul.is_none());
        assert!(product.final_reduce.is_empty());
    }
}
