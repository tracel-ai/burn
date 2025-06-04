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
    last_merged_failed: bool,
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
            last_merged_failed: false,
        }
    }
    pub fn register(&mut self, operation: &OperationIr) {
        // println!("Register {operation:?}");
        if self.last_merged_failed {
            return;
        }

        match self.merge_blocks(operation) {
            MergeBlockStep::Full => {
                println!("Full merge");
            }
            MergeBlockStep::NoNeed => {
                println!("No Need");
            }
            MergeBlockStep::Fail | MergeBlockStep::Partial => {
                // With the given operation, blocks are no longer independent.
                println!("Merge fail");
                self.last_merged_failed = true;
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
        self.last_merged_failed = false;
    }

    pub fn still_optimizing(&self) -> bool {
        if self.last_merged_failed {
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

        println!("{:?}", self.blocks);
        println!("Merge blocks {block_merges:?}");
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
