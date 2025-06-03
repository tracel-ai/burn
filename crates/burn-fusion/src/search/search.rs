use burn_ir::OperationIr;

use crate::{NumOperations, OptimizationBuilder, stream::store::ExecutionStrategy};

use super::{
    Block, Registration,
    merging::{MergeBlockResult, merge_blocks},
    searcher::Searcher,
};

pub struct OptimizationSearch<O> {
    builders: Vec<Box<dyn OptimizationBuilder<O>>>,
    blocks: Vec<Block<O>>,
    length: usize,
    stop: bool,
}

#[derive(Debug)]
pub struct OptimizationSearchResult<O> {
    pub strategy: ExecutionStrategy<O>,
    pub num_operations: usize,
    pub ordering: Vec<usize>,
}

impl<O: NumOperations> OptimizationSearch<O> {
    pub fn new(builders: Vec<Box<dyn OptimizationBuilder<O>>>) -> Self {
        Self {
            builders,
            blocks: Vec::new(),
            length: 0,
            stop: false,
        }
    }
    pub fn register(&mut self, operation: &OperationIr) {
        if self.stop {
            return;
        }

        match self.merge_blocks(operation) {
            MergeBlockStep::Full | MergeBlockStep::NoNeed => {}
            MergeBlockStep::Fail | MergeBlockStep::Partial => {
                // With the given operation, blocks are no longer independent.
                self.stop = true;
                return;
            }
        }

        let mut added_count = 0;

        for block in self.blocks.iter_mut() {
            match block.register(operation, false, self.length) {
                Registration::Accepted => {
                    added_count += 1;
                }
                Registration::NotPartOfTheGraph => {}
            }
        }
        if added_count == 0 {
            self.on_new_block(operation);
        } else {
            assert_eq!(added_count, 1, "Can only add the operation to one block.");
        }
        self.length += 1;
    }

    pub fn execute(&self) -> OptimizationSearchResult<O> {
        Searcher::new(self.blocks.clone()).search()
    }

    pub fn reset(&mut self) {
        self.builders.iter_mut().for_each(|b| b.reset());
        self.length = 0;
        self.blocks.clear();
        self.stop = false;
    }

    pub fn still_optimizing(&self) -> bool {
        if self.stop {
            return false;
        }
        if self.blocks.is_empty() {
            return true;
        }

        let mut num_stopped = 0;

        for block in self.blocks.iter() {
            if !block.still_optimizing() {
                num_stopped += 1
            }
        }

        num_stopped < self.blocks.len()
    }

    fn merge_blocks(&mut self, operation: &OperationIr) -> MergeBlockStep {
        let nodes = operation.nodes();
        let mut block_merges = Vec::new();

        for (i, block) in self.blocks.iter().enumerate() {
            if block.should_include_nodes(&nodes) {
                block_merges.push(i);
            }
        }

        if block_merges.len() <= 1 {
            return MergeBlockStep::NoNeed;
        }

        let blocks_to_merge = self
            .blocks
            .iter()
            .enumerate()
            .filter_map(|(i, g)| match block_merges.contains(&i) {
                true => Some(g),
                false => None,
            })
            .collect::<Vec<_>>();
        let merged = merge_blocks(&blocks_to_merge, false);
        println!("Merge block during register. {blocks_to_merge:?} => {merged:?}");

        let mut clear_blocks = || {
            println!("Clear blocks from {:?}", self.blocks);
            let mut indices = block_merges.to_vec();
            indices.sort();

            for g in indices.into_iter().rev() {
                self.blocks.remove(g);
            }
            println!("Clear blocks into {:?}", self.blocks);
        };

        match merged {
            MergeBlockResult::Full(block) => {
                clear_blocks();
                self.blocks.push(block);
                Block::sort(&mut self.blocks);
                println!("Full {:?}", self.blocks);
                MergeBlockStep::Full
            }
            MergeBlockResult::Partial {
                mut merged,
                mut failed,
            } => {
                clear_blocks();
                self.blocks.append(&mut merged);
                self.blocks.append(&mut failed);
                Block::sort(&mut self.blocks);
                println!("Partial {:?}", self.blocks);
                MergeBlockStep::Partial
            }
            MergeBlockResult::Fail => MergeBlockStep::Fail,
        }
    }

    fn on_new_block(&mut self, operation: &OperationIr) {
        let mut block = Block::new(&self.builders);
        block.register(operation, true, self.length);
        self.blocks.push(block);
    }
}

enum MergeBlockStep {
    Full,
    Partial,
    Fail,
    NoNeed,
}
