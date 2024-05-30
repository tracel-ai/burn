use super::{Operation, Procedure, Variable};
use crate::ir::ReadGlobalWithLayout;

/// Information necessary when compiling a scope.
pub struct ScopeProcessing {
    /// The variable declarations.
    pub variables: Vec<Variable>,
    /// The operations.
    pub operations: Vec<Operation>,
}

impl ScopeProcessing {
    /// Optimize the [variables](Variable) and [operations](Operation).
    ///
    /// ## Notes:
    ///
    /// This should be called once right after the creation of the type.
    /// If you built this type from the [scope process function](super::Scope::process), you don't have to
    /// call it again.
    pub fn optimize(self) -> Self {
        self.merge_read_global_with_layout()
    }

    /// Merge all compatible [read global with layout procedures](ReadGlobalWithLayout).
    fn merge_read_global_with_layout(mut self) -> Self {
        #[derive(Default)]
        struct Optimization {
            merged_procs: Vec<MergedProc>,
        }

        #[derive(new)]
        struct MergedProc {
            proc: ReadGlobalWithLayout,
            positions: Vec<usize>,
        }

        impl Optimization {
            fn new(existing_operations: &[Operation]) -> Self {
                let mut optim = Self::default();

                existing_operations
                    .iter()
                    .enumerate()
                    .for_each(|(position, operation)| {
                        if let Operation::Procedure(Procedure::ReadGlobalWithLayout(proc)) =
                            operation
                        {
                            optim.register_one(proc, position);
                        }
                    });

                optim
            }

            fn register_one(&mut self, proc: &ReadGlobalWithLayout, position: usize) {
                for merged_proc in self.merged_procs.iter_mut() {
                    if let Some(merged) = merged_proc.proc.try_merge(proc) {
                        merged_proc.proc = merged;
                        merged_proc.positions.push(position);
                        return;
                    }
                }

                self.merged_procs
                    .push(MergedProc::new(proc.clone(), vec![position]));
            }

            fn apply(self, existing_operations: Vec<Operation>) -> Vec<Operation> {
                if self.merged_procs.is_empty() {
                    return existing_operations;
                }

                let mut operations = Vec::with_capacity(existing_operations.len());

                for (position, operation) in existing_operations.into_iter().enumerate() {
                    let mut is_merged_op = false;

                    for merged_proc in self.merged_procs.iter() {
                        if merged_proc.positions[0] == position {
                            operations.push(Operation::Procedure(Procedure::ReadGlobalWithLayout(
                                merged_proc.proc.clone(),
                            )));
                            is_merged_op = true;
                        }

                        if merged_proc.positions.contains(&position) {
                            is_merged_op = true;
                        }
                    }

                    if !is_merged_op {
                        operations.push(operation);
                    }
                }

                operations
            }
        }

        let optimization = Optimization::new(&self.operations);
        self.operations = optimization.apply(self.operations);
        self
    }
}
