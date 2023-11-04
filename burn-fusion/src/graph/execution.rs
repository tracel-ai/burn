use crate::{Handle, HandleContainer};

use super::{FusedBackend, FusionStatus, Graph, Optimization};

pub trait GraphExecution<B: FusedBackend>: Default + Send {
    /// Maybe execute the given graph using the list of potential operations.
    fn maybe_execute(
        &mut self,
        graph: &mut Graph<B>,
        handles: &mut HandleContainer<B>,
        optimizations: &mut [Optimization<B>],
        force: bool,
    );
}

#[derive(Default)]
pub struct GreedyGraphExecution;

impl<B: FusedBackend> GraphExecution<B> for GreedyGraphExecution {
    fn maybe_execute(
        &mut self,
        graph: &mut Graph<B>,
        handles: &mut HandleContainer<B>,
        optimizations: &mut [Optimization<B>],
        force: bool,
    ) {
        loop {
            if !force && still_optimizing(optimizations) {
                break;
            }

            match find_best_optimization_index(optimizations) {
                Some(index) => graph.execute_optimization(handles, index, optimizations),
                None => graph.execute(handles),
            }

            if graph.is_empty() {
                // No more ops to fuse.
                break;
            }
        }
    }
}

fn still_optimizing<B: FusedBackend>(optimizations: &[Optimization<B>]) -> bool {
    let mut num_stopped = 0;

    for optimization in optimizations.iter() {
        match optimization.status {
            FusionStatus::Closed(_) => num_stopped += 1,
            _ => {}
        };
    }

    num_stopped < optimizations.len()
}

fn find_best_optimization_index<B: FusedBackend>(
    optimizations: &[Optimization<B>],
) -> Option<usize> {
    let mut best_index = None;
    let mut best_score = 0;

    for (i, optimization) in optimizations.iter().enumerate() {
        let properties = match optimization.status {
            FusionStatus::Closed(properties) => properties,
            FusionStatus::Open(properties) => properties,
        };

        if properties.ready && properties.score >= best_score {
            best_index = Some(i);
            best_score = properties.score;
        }
    }

    best_index
}
