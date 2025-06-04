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
    stopped: bool,
    max_blocks: Option<usize>,
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
            stopped: false,
            max_blocks: None,
        }
    }

    pub fn max_blocks(&mut self, max_blocks: usize) {
        self.max_blocks = Some(max_blocks);
    }

    pub fn register(&mut self, operation: &OperationIr) {
        if self.stopped {
            return;
        }

        if self.blocks.is_empty() {
            self.on_new_block(operation);
            self.length += 1;
            return;
        }

        match self.merge_blocks(operation, false) {
            MergeBlockStep::Full => {}
            MergeBlockStep::NoNeed => {}
            MergeBlockStep::Fail | MergeBlockStep::Partial => {
                // With the given operation, blocks are no longer independent.
                self.stopped = true;
                return;
            }
        }

        if let Some(max_blocks) = self.max_blocks {
            if max_blocks == 1 {
                self.register_inner(operation, true);
            } else {
                let added_count = self.register_inner(operation, false);

                if added_count == 0 && self.blocks.len() < max_blocks {
                    self.on_new_block(operation);
                } else {
                    self.stopped = true;
                    return;
                    self.merge_blocks(operation, true);

                    if self.blocks.len() >= max_blocks {
                        self.stopped = true;
                    } else {
                        let added_count = self.register_inner(operation, false);
                        if added_count == 0 {
                            self.on_new_block(operation);
                        }
                    }
                }
            }
        } else {
            let added_count = self.register_inner(operation, false);
            if added_count == 0 {
                self.on_new_block(operation);
            }
        }

        self.length += 1;
    }

    fn register_inner(&mut self, operation: &OperationIr, force: bool) -> usize {
        let mut added_count = 0;
        for block in self.blocks.iter_mut() {
            match block.register(operation, force, self.length) {
                Registration::Accepted => {
                    added_count += 1;
                }
                Registration::NotPartOfTheGraph => {}
            }
        }
        added_count
    }

    pub fn execute(&self, operations: &[OperationIr]) -> OptimizationSearchResult<O> {
        let result = Searcher::new(self.blocks.clone()).search(operations, || self.empty_search());

        if result.num_operations == 1 {
            let mut search = self.empty_search();
            search.max_blocks(1);
            for op in operations.iter() {
                search.register(op);
                if !search.still_optimizing() {
                    break;
                }
            }

            return search.execute_no_recurse(operations);
        }

        result
    }

    pub fn execute_no_recurse(&self, operations: &[OperationIr]) -> OptimizationSearchResult<O> {
        Searcher::new(self.blocks.clone()).search(operations, || self.empty_search())
    }

    fn empty_search(&self) -> Self {
        Self::new(
            self.builders
                .iter()
                .map(|b| {
                    let mut b = b.clone_dyn();
                    b.reset();
                    b
                })
                .collect(),
        )
    }

    pub fn reset(&mut self) {
        self.builders.iter_mut().for_each(|b| b.reset());
        self.length = 0;
        self.blocks.clear();
        self.stopped = false;
    }

    pub fn still_optimizing(&self) -> bool {
        if self.stopped {
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

    fn merge_blocks(&mut self, operation: &OperationIr, all: bool) -> MergeBlockStep {
        let nodes = operation.nodes();
        let mut block_merges = Vec::new();

        for (i, block) in self.blocks.iter().enumerate() {
            if all || block.should_include_nodes(&nodes) {
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

        let mut clear_blocks = || {
            let mut indices = block_merges.to_vec();
            indices.sort();

            for g in indices.into_iter().rev() {
                self.blocks.remove(g);
            }
        };

        match merged {
            MergeBlockResult::Full(block) => {
                clear_blocks();
                self.blocks.push(block);
                Block::sort(&mut self.blocks);
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
                MergeBlockStep::Partial
            }
            MergeBlockResult::Fail => MergeBlockStep::Fail,
        }
    }

    fn on_new_block(&mut self, operation: &OperationIr) {
        let mut block = Block::new(&self.builders);
        block.register(operation, true, self.length);
        self.blocks.push(block);
        log::info!("New blocks {:?}", self.blocks.len());
    }
}

enum MergeBlockStep {
    Full,
    Partial,
    Fail,
    NoNeed,
}
