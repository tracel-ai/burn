use crate::engine::{
    codegen::ir::{FuseArg, FuseOp, UnaryFuseArgs},
    trace::FuseTrace,
};
use burn_ir::OperationIr;

#[derive(Debug, Clone, Default)]
/// Tracks and evaluates the efficiency of operation fusion.
///
/// This struct computes a "fusion score" by comparing the total number of
/// global memory reads and writes required by individual (unfused) operations
/// against the actual I/O performed by a fused kernel trace.
pub struct Scoring {
    num_writes: usize,
    num_reads: usize,
}

impl Scoring {
    /// Resets the internal I/O counters to zero.
    pub fn reset(&mut self) {
        self.num_writes = 0;
        self.num_reads = 0;
    }

    /// Registers an unfused operation to the score, counting its total potential I/O.
    pub fn register(&mut self, op: &OperationIr) {
        self.num_writes += op.outputs().count();
        self.num_reads += op.inputs().count();
    }

    /// Evaluates the efficiency of a fused trace by comparing its actual I/O
    /// against the registered unfused I/O. Returns the number of saved I/O operations.
    pub fn evaluate(&self, trace: &FuseTrace) -> u64 {
        let mut num_reads_fused = 0;
        let mut num_writes_fused = 0;

        for b in trace.blocks.iter() {
            // Count reads in block
            for (_, ops) in b.reads.iter() {
                num_reads_fused += self.count_fused_io(ops, |args| &args.input);
            }
            // Count writes in block
            for (_, ops) in b.writes.iter() {
                num_writes_fused += self.count_fused_io(ops, |args| &args.out);
            }
        }

        self.calculate_score(num_reads_fused, num_writes_fused)
    }

    fn calculate_score(&self, reads_fused: usize, writes_fused: usize) -> u64 {
        let num_fused = reads_fused + writes_fused;
        let num_unfused = self.num_reads + self.num_writes;

        if num_fused >= num_unfused {
            0
        } else {
            (num_unfused - num_fused) as u64
        }
    }

    fn count_fused_io<F>(&self, ops: &[FuseOp], arg_extractor: F) -> usize
    where
        F: Fn(&UnaryFuseArgs) -> &FuseArg,
    {
        ops.iter()
            .filter(|op| {
                let FuseOp::Assign(args) = op else {
                    unreachable!()
                };
                matches!(
                    arg_extractor(args),
                    FuseArg::Input(..)
                        | FuseArg::InputReshaped { .. }
                        | FuseArg::InputSwapDims { .. }
                        | FuseArg::Output(..)
                )
            })
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scoring_savings() {
        let mut scoring = Scoring::default();
        // Simulate 10 unfused I/O operations registered
        scoring.num_reads = 5;
        scoring.num_writes = 5;

        // If fused trace only uses 4 I/O operations, we saved 6
        let score = scoring.calculate_score(2, 2);
        assert_eq!(score, 6);
    }

    #[test]
    fn test_scoring_no_savings() {
        let mut scoring = Scoring::default();
        scoring.num_reads = 2;
        scoring.num_writes = 2;

        // If fused I/O is equal or worse, return 0
        let score = scoring.calculate_score(3, 3);
        assert_eq!(score, 0);
    }

    #[test]
    fn test_reset() {
        let mut scoring = Scoring {
            num_writes: 10,
            num_reads: 10,
        };
        scoring.reset();
        assert_eq!(scoring.num_writes, 0);
        assert_eq!(scoring.num_reads, 0);
    }
}
