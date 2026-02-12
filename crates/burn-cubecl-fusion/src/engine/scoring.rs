use crate::engine::{
    codegen::ir::{FuseArg, FuseOp, UnaryFuseArgs},
    trace::FuseTrace,
};
use burn_ir::OperationIr;

#[derive(Debug, Clone, Default)]
/// Tracks and evaluates the efficiency of operation fusion.
pub struct Scoring {
    num_writes: usize,
    num_reads: usize,
    num_ops: usize,
}

impl Scoring {
    /// Resets the internal O counters.
    pub fn reset(&mut self) {
        self.num_writes = 0;
        self.num_reads = 0;
        self.num_ops = 0;
    }

    /// Registers an unfused operation to the score, counting its total potential I/O.
    pub fn register(&mut self, op: &OperationIr) {
        self.num_writes += op.outputs().count();
        self.num_reads += op.inputs().count();
        self.num_ops += 1;
    }

    /// Evaluates the efficiency of a fused trace by comparing its actual I/O
    /// against the registered unfused I/O. Returns the number of saved I/O operations.
    pub fn evaluate(&self, trace: &FuseTrace) -> u64 {
        let mut num_reads_fused = 0;
        let mut num_writes_fused = 0;
        let mut num_penalty = 0;

        for b in trace.blocks.iter() {
            // Count reads in block
            for (_, ops) in b.reads.iter() {
                let result = self.count_fused_io(ops, |args| &args.input);
                num_reads_fused += result.0;
                num_penalty += result.1;
            }
            // Count writes in block
            for (_, ops) in b.writes.iter() {
                let result = self.count_fused_io(ops, |args| &args.out);
                num_writes_fused += result.0;
                num_penalty += result.1;
            }
        }

        self.calculate_score(num_reads_fused, num_writes_fused, num_penalty)
    }

    fn calculate_score(&self, reads_fused: usize, writes_fused: usize, num_penalty: usize) -> u64 {
        // Those could be tweaked eventually.

        const FACTOR_IO: u64 = 100;
        const FACTOR_LAUNCH: u64 = 10;
        const FACTOR_PENALTY: u64 = 150;

        let num_fused = reads_fused + writes_fused;
        let num_unfused = self.num_reads + self.num_writes;

        let score_io = match num_fused >= num_unfused {
            true => 0,
            false => (num_unfused - num_fused) as u64 * FACTOR_IO,
        };

        // We minus 1 since at least one kernel launch is necessary.
        let score_launch = self.num_ops.checked_sub(1).unwrap_or(0) as u64 * FACTOR_LAUNCH;

        let score_penalty = num_penalty as u64 * FACTOR_PENALTY;

        (score_io + score_launch)
            .checked_sub(score_penalty)
            .unwrap_or(0)
    }

    fn count_fused_io<F>(&self, ops: &[FuseOp], arg_extractor: F) -> (usize, usize)
    where
        F: Fn(&UnaryFuseArgs) -> &FuseArg,
    {
        let mut num_io = 0;
        let mut penalty = 0;

        for op in ops.iter() {
            let FuseOp::Assign(args) = op else {
                unreachable!()
            };
            let count_normal = matches!(
                arg_extractor(args),
                FuseArg::Input(..) | FuseArg::Output(..)
            ) as usize;
            let count_view = matches!(
                arg_extractor(args),
                FuseArg::InputReshaped { .. } | FuseArg::InputSwapDims { .. }
            ) as usize;
            num_io += count_normal + count_view;
            penalty += count_view;
        }

        (num_io, penalty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scoring_io_savings() {
        let mut scoring = Scoring::default();
        // 2 ops: each 1 read, 1 write = 4 total IO, 2 ops
        scoring.num_reads = 2;
        scoring.num_writes = 2;
        scoring.num_ops = 2;

        // Fused: 1 read, 1 write, 0 penalty
        // IO saved: (4 - 2) * 100 = 200
        // Launch saved: (2 - 1) * 10 = 10
        // Total = 210
        let score = scoring.calculate_score(1, 1, 0);
        assert_eq!(score, 210);
    }

    #[test]
    fn test_scoring_with_penalties() {
        let mut scoring = Scoring::default();
        // 2 ops: 4 total IO, 2 ops
        scoring.num_reads = 2;
        scoring.num_writes = 2;
        scoring.num_ops = 2;

        // Fused: 1 read, 1 write, BUT 1 penalty (e.g., a Reshape)
        // IO saved: 200
        // Launch saved: 10
        // Penalty: 1 * 150 = 150
        // Total = (200 + 10) - 150 = 60
        let score = scoring.calculate_score(1, 1, 1);
        assert_eq!(score, 60);
    }

    #[test]
    fn test_penalty_outweighs_benefit() {
        let mut scoring = Scoring::default();
        scoring.num_reads = 1;
        scoring.num_writes = 1;
        scoring.num_ops = 2;

        // If penalty is high (150) and we only saved a launch (10)
        // (0 IO saved + 10 Launch) - 150 = -140 -> clamped to 0
        let score = scoring.calculate_score(1, 1, 1);
        assert_eq!(score, 0);
    }

    #[test]
    fn test_scoring_no_ops() {
        let scoring = Scoring::default();
        // Should not panic with 0 ops due to checked_sub
        let score = scoring.calculate_score(0, 0, 0);
        assert_eq!(score, 0);
    }

    #[test]
    fn test_reset() {
        let mut scoring = Scoring {
            num_writes: 10,
            num_reads: 10,
            num_ops: 10,
        };
        scoring.reset();
        assert_eq!(scoring.num_writes, 0);
        assert_eq!(scoring.num_reads, 0);
        assert_eq!(scoring.num_ops, 0);
    }
}
