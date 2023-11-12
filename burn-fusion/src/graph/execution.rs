use super::{Graph, Optimization};
use crate::{FusionBackend, FusionStatus, HandleContainer};

/// The graph execution trait abstracts the way the graph is executing optimizations.
pub trait GraphExecution<B: FusionBackend>: Default + Send {
    /// Execute the given graph using the list of potential [optimizations](Optimization).
    /// May do nothing if empty or not ready
    fn maybe_execute(
        &mut self,
        graph: &mut Graph<B>,
        handles: &mut HandleContainer<B>,
        optimizations: &mut [Optimization<B>],
        force: bool,
    );
}

/// Execute an optimization following a greedy algorithm.
#[derive(Default)]
pub struct GreedyGraphExecution;

impl<B: FusionBackend> GraphExecution<B> for GreedyGraphExecution {
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

fn still_optimizing<B: FusionBackend>(optimizations: &[Optimization<B>]) -> bool {
    let mut num_stopped = 0;

    for optimization in optimizations.iter() {
        if let FusionStatus::Closed(_) = optimization.status {
            num_stopped += 1
        }
    }

    num_stopped < optimizations.len()
}

fn find_best_optimization_index<B: FusionBackend>(
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
