use crate::{OptimizerStateMapper, RecordState, StatePath};

/// State whose floating tensors can be transformed, including nested and optional components.
/// Implementations visit every tensor exactly once and preserve scalar bookkeeping.
pub trait StateMappable: RecordState + Clone {
    /// Apply a mapper to each tensor leaf while preserving the structure of the state.
    fn map_tensors<M: OptimizerStateMapper>(self, mapper: &M) -> Self {
        self.map_tensors_at(mapper, &StatePath::Root)
    }

    /// Visit this component beneath an existing record path, extending it for each field.
    fn map_tensors_at<M: OptimizerStateMapper>(self, mapper: &M, path: &StatePath<'_>) -> Self;
}

impl StateMappable for () {
    fn map_tensors_at<M: OptimizerStateMapper>(self, _mapper: &M, _path: &StatePath<'_>) -> Self {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};
    use burn::tensor::{Device, Tensor};
    use burn_core as burn;

    #[derive(Clone, RecordState, crate::StateMappable)]
    struct Component<const D: usize> {
        velocity: Tensor<D>,
    }

    #[derive(Clone, RecordState, crate::StateMappable)]
    struct TwinState<const D: usize> {
        pub left: Component<D>,
        pub right: Component<D>,
        history: Vec<Tensor<1>>,
    }

    #[derive(Clone)]
    struct ComponentStorage;
    impl OptimizerStateMapper for ComponentStorage {
        fn map<const D: usize>(&self, tensor: Tensor<D>, path: &StatePath<'_>) -> Tensor<D> {
            if path == &StatePath::Root.field("history").index(1)
                || path.matches(&["right", "velocity"])
            {
                tensor.cast(burn::tensor::FloatDType::BF16)
            } else {
                tensor
            }
        }
    }

    #[test]
    fn traversal_distinguishes_repeated_fields_and_sequence_elements() {
        use burn::tensor::DType;
        let device = Device::default();
        let tensor = Tensor::<1>::ones([2], &device);
        let state = TwinState::<1> {
            left: Component {
                velocity: tensor.clone(),
            },
            right: Component {
                velocity: tensor.clone(),
            },
            history: vec![tensor.clone(), tensor],
        }
        .map_tensors(&ComponentStorage);
        assert_eq!(state.left.velocity.dtype(), DType::F32);
        assert_eq!(state.right.velocity.dtype(), DType::BF16);
        assert_eq!(state.history[0].dtype(), DType::F32);
        assert_eq!(state.history[1].dtype(), DType::BF16);
    }
}
