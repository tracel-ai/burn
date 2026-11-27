use smallvec::smallvec;

use crate::{
    OpSharding,
    Sharding::{self, Replicated},
};

/// A 2D grid sample reads each channel of each batch entry whole, at the points its grid gives
/// that entry: a split of the batch splits the grid alike, one of the channels leaves the grid
/// whole, and anything else gathers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridSampleRule {
    input: Sharding,
    grid: Sharding,
}

impl GridSampleRule {
    pub fn new(input: Sharding) -> Self {
        match input.split_dim() {
            Some(0) => Self { input, grid: input },
            Some(1) => Self {
                input,
                grid: Replicated,
            },
            _ => Self {
                input: Replicated,
                grid: Replicated,
            },
        }
    }

    pub fn sharding(&self) -> OpSharding {
        OpSharding {
            inputs: smallvec![self.input, self.grid],
            output: self.input,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Sharding::Split;

    #[test]
    fn a_split_of_the_batch_splits_the_grid_and_one_of_the_channels_does_not() {
        assert_eq!(
            GridSampleRule::new(Split { dim: 0 }).sharding().inputs[..],
            [Split { dim: 0 }; 2]
        );
        assert_eq!(
            GridSampleRule::new(Split { dim: 1 }).sharding().inputs[..],
            [Split { dim: 1 }, Replicated]
        );
    }

    #[test]
    fn a_split_of_the_image_gathers() {
        assert_eq!(
            GridSampleRule::new(Split { dim: 2 }).sharding().output,
            Replicated
        );
    }
}
