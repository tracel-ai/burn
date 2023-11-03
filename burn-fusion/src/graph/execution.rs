use super::{FusedBackend, FusionStatus, Graph, HandleContainer, Optimization};

pub trait GraphExecution<B: FusedBackend> {
    /// Maybe execute the given graph using the list of potential operations.
    fn maybe_execute(
        &mut self,
        graph: &mut Graph<B>,
        handles: &mut HandleContainer<B::Handle>,
        optimizations: &mut [Optimization<B>],
    );
}

pub struct GreedyGraphExecution;

impl<B: FusedBackend> GraphExecution<B> for GreedyGraphExecution {
    fn maybe_execute(
        &mut self,
        graph: &mut Graph<B>,
        handles: &mut HandleContainer<B::Handle>,
        optimizations: &mut [Optimization<B>],
    ) {
        loop {
            let mut num_stopped = 0;

            for optimization in optimizations.iter() {
                match optimization.status {
                    FusionStatus::Closed(_) => num_stopped += 1,
                    _ => {}
                };
            }

            if num_stopped < optimizations.len() {
                // not executing, some are still fusing.
                break;
            }

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

            match best_index {
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
