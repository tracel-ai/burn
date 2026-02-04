use super::block::ReduceBlockKind;
use crate::optim::reduce_broadcasted::fuser::block::ReduceBlockFuser;
use burn_ir::{TensorId, TensorIr};
use cubecl::Runtime;
use std::collections::BTreeMap;

#[derive(Debug)]
pub struct FullFuserAnalyzer {
    // We need to know the block id of which we can reuse the read local input.
    analyses: Vec<Vec<(TensorIr, usize)>>,
}

impl FullFuserAnalyzer {
    pub fn new<R: Runtime>(blocks: &[ReduceBlockFuser<R>]) -> Self {
        let mut state = AnalysisState::default();

        for block in blocks.iter() {
            for (pos, op) in block.ops.iter().enumerate() {
                let potential_from_previous_blocks = op.inputs();
                let potential_to_next_blocks = op.outputs();

                match &block.kind {
                    ReduceBlockKind::Elemwise => {
                        state.register(
                            potential_from_previous_blocks,
                            potential_to_next_blocks,
                            BlockKind::Full,
                        );
                    }
                    ReduceBlockKind::Reduce { ops_index, .. } => {
                        if pos < *ops_index {
                            state.register(
                                potential_from_previous_blocks,
                                potential_to_next_blocks,
                                BlockKind::Full,
                            );
                        } else if pos > *ops_index {
                            state.register(
                                potential_from_previous_blocks,
                                potential_to_next_blocks,
                                BlockKind::Single,
                            );
                        } else {
                            state.next_block();
                        }
                    }
                }
            }
            state.next_block();
        }

        // First one is never called.
        state.analyses.remove(0);

        Self {
            analyses: state.analyses,
        }
    }

    pub fn retrieve_next(&mut self) -> FullFuserAnalysis {
        let inputs = self.analyses.remove(0);
        FullFuserAnalysis { inputs }
    }
}

pub struct FullFuserAnalysis {
    /// The tensor received from a previous block.
    pub inputs: Vec<(TensorIr, usize)>,
}

#[derive(Default)]
struct AnalysisState {
    /// That pool contains tensors that are available in the fuse-on-write part of a reduce, not
    /// broadcasted.
    available_from_previous_single: BTreeMap<TensorId, usize>,
    /// That pool contains tensors that are available in the fuse-on-read of a reduce and the
    /// element-wise broadcasted part
    available_from_previous_full: BTreeMap<TensorId, usize>,
    // With the trace I get get the fuse arg.
    block_data: Vec<(TensorIr, usize)>,
    analyses: Vec<Vec<(TensorIr, usize)>>,
    current_full: Vec<TensorIr>,
    current_single: Vec<TensorIr>,
}

enum BlockKind {
    Full,
    Single,
}

impl AnalysisState {
    fn next_block(&mut self) {
        let block_pos = self.analyses.len();
        let data = core::mem::take(&mut self.block_data);
        self.analyses.push(data);

        // Makes the current tensor reads available for the next block.
        for p in self.current_single.drain(..) {
            // We need to keep the earliest block position.
            self.available_from_previous_single
                .entry(p.id)
                .or_insert(block_pos);
        }
        for p in self.current_full.drain(..) {
            // We need to keep the earliest block position.
            self.available_from_previous_full
                .entry(p.id)
                .or_insert(block_pos);
        }
    }

    fn register<'a>(
        &mut self,
        potential_from_previous_blocks: impl Iterator<Item = &'a TensorIr>,
        potential_to_next_blocks: impl Iterator<Item = &'a TensorIr>,
        kind: BlockKind,
    ) {
        match kind {
            BlockKind::Full => {
                for potential in potential_from_previous_blocks {
                    // We can't since it's not in the same scope.
                    //
                    // TODO: Find a way to merge multiple reduce loops.
                    //
                    // if let Some(block_pos) = self.available_from_previous_full.get(&potential.id) {
                    //     self.block_data.push((potential.clone(), *block_pos));
                    // }

                    // We can since it's a broadcast.
                    if let Some(block_pos) = self.available_from_previous_single.get(&potential.id)
                    {
                        self.block_data.push((potential.clone(), *block_pos));
                    }

                    // Can reuse the read.
                    self.current_full.push(potential.clone());
                }

                for p in potential_to_next_blocks {
                    self.current_full.push(p.clone());
                }
            }
            BlockKind::Single => {
                for potential in potential_from_previous_blocks {
                    if let Some(block_pos) = self.available_from_previous_single.get(&potential.id)
                    {
                        self.block_data.push((potential.clone(), *block_pos));
                    }
                    // Can reuse the read.
                    self.current_single.push(potential.clone());
                }

                for p in potential_to_next_blocks {
                    self.current_single.push(p.clone());
                }
            }
        }
    }
}
