use burn_cube::dialect::{Item, Scope, Variable};

use crate::JitElement;

/// Specifies the reduce dim algorithm in use
pub trait ReduceDimNaive<E: JitElement>: Send + Sync + 'static {
    /// The reduction accumulator
    type Accumulator: Copy;

    /// Initialization for naive algorithm
    fn initialize_naive(
        scope: &mut Scope,
        input_item: Item,
        output_item: Item,
    ) -> Self::Accumulator;

    /// Inner loop for naive algorithm
    fn inner_loop_naive(
        scope: &mut Scope,
        accumulator: Self::Accumulator,
        current_value: Variable,
        i: Variable,
    );

    /// Assignation for naive algorithm
    fn assign_naive(
        scope: &mut Scope,
        output: Variable,
        accumulator: Self::Accumulator,
        shape_reduce_dim: Variable,
    );
}
