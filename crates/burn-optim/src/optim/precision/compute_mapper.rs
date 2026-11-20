use crate::OptimizerStateMapper;
use burn_core::tensor::{FloatDType, Tensor};

pub(super) struct ComputeOptimizerStateMapper {
    dtype: FloatDType,
}
impl ComputeOptimizerStateMapper {
    pub(super) fn new(dtype: FloatDType) -> Self {
        Self { dtype }
    }
}
impl OptimizerStateMapper for ComputeOptimizerStateMapper {
    fn map<const D: usize>(&mut self, tensor: Tensor<D>, _name: &str) -> Tensor<D> {
        tensor.cast(self.dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RecordState, StateMappable};
    use burn::tensor::Device;
    use burn_core as burn;

    #[derive(Clone, RecordState)]
    struct MixedRankState {
        vector: Tensor<1>,
        matrix: Tensor<2>,
        steps: usize,
    }
    impl StateMappable for MixedRankState {
        fn map_tensors<M: OptimizerStateMapper>(self, mapper: &mut M) -> Self {
            Self {
                vector: mapper.map(self.vector, "MixedRankState.vector"),
                matrix: mapper.map(self.matrix, "MixedRankState.matrix"),
                steps: self.steps,
            }
        }
    }

    #[test]
    fn state_traversal_preserves_auxiliary_ranks_and_counters() {
        let device = Device::default();
        let state = MixedRankState {
            vector: Tensor::from_floats([1.003], &device),
            matrix: Tensor::from_floats([[2.003, 3.003]], &device),
            steps: 7,
        };
        let state = state.map_tensors(&mut ComputeOptimizerStateMapper::new(FloatDType::BF16));
        assert_eq!(state.vector.into_scalar::<f32>(), 1.0);
        assert_eq!(state.matrix.dims(), [1, 2]);
        assert_eq!(state.matrix.try_into_vec_as::<f32>().unwrap(), [2.0, 3.0]);
        assert_eq!(state.steps, 7);
    }
}
