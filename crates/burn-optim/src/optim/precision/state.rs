use burn_core::tensor::Tensor;

use crate::RecordState;

/// Storage constraints declared by an optimizer state component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateTensorKind {
    /// Signed statistics, such as a first moment or momentum buffer.
    Signed,
    /// Non-negative statistics, such as an exponential average of squared gradients.
    NonNegative,
    /// A tensor whose invariants require retaining the computation precision.
    ///
    /// Examples include running maxima and statistics used in cancellation-sensitive estimates.
    FullPrecision,
}

/// An on-device transformation of optimizer state tensors.
///
/// Implementations must preserve shape and device. A component's storage constraints apply when
/// encoding persistent state; decoding prepares all tensors for the optimizer's arithmetic.
pub trait OptimizerStateMapper {
    /// Transform one tensor according to the owning component's storage constraints.
    fn map<const D: usize>(&mut self, tensor: Tensor<D>, kind: StateTensorKind) -> Tensor<D>;
}

/// Optimizer state that can be transformed without serialization or host readback.
///
/// Visit every floating tensor exactly once, including nested and optional state. Keep scalars
/// unchanged. Custom state components should use [`StateTensorKind::FullPrecision`] unless a
/// lower-precision representation is appropriate for their numerical invariants.
pub trait OptimizerState: RecordState + Clone {
    /// Transform the tensors owned by this state, returning the updated state.
    fn map_tensors<M: OptimizerStateMapper>(self, mapper: &mut M) -> Self;
}

impl OptimizerState for () {
    fn map_tensors<M: OptimizerStateMapper>(self, _mapper: &mut M) -> Self {}
}

pub(crate) trait MapStateTensors: Sized {
    fn map<M: OptimizerStateMapper>(self, mapper: &mut M, kind: StateTensorKind) -> Self;
}

impl<const D: usize> MapStateTensors for Tensor<D> {
    fn map<M: OptimizerStateMapper>(self, mapper: &mut M, kind: StateTensorKind) -> Self {
        mapper.map(self, kind)
    }
}

impl<T: MapStateTensors> MapStateTensors for Option<T> {
    fn map<M: OptimizerStateMapper>(self, mapper: &mut M, kind: StateTensorKind) -> Self {
        self.map(|tensor| tensor.map(mapper, kind))
    }
}

// Declare numerical constraints beside the component; generate only the structural traversal.
macro_rules! impl_optimizer_state {
    ($name:ident {
        $(tensors { $($field:ident: $kind:ident),* $(,)? })?
        $(nested { $($nested:ident),* $(,)? })?
        $(optional { $($optional:ident),* $(,)? })?
    }) => {
        impl<const D: usize> crate::OptimizerState for $name<D> {
            fn map_tensors<M: crate::OptimizerStateMapper>(mut self, mapper: &mut M) -> Self {
                $($(
                    self.$field = crate::optim::precision::state::MapStateTensors::map(
                        self.$field, mapper, crate::StateTensorKind::$kind,
                    );
                )*)?
                $($(
                    self.$nested = crate::OptimizerState::map_tensors(self.$nested, mapper);
                )*)?
                $($(
                    self.$optional = self.$optional.map(|state|
                        crate::OptimizerState::map_tensors(state, mapper)
                    );
                )*)?
                self
            }
        }
    };
}

pub(crate) use impl_optimizer_state;
