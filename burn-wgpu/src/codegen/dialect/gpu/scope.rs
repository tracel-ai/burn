use super::{
    Algorithm, Elem, Item, Operation, Operator, ReadGlobalAlgo, ReadGlobalWithLayoutAlgo,
    UnaryOperator, Variable, Vectorization, WriteGlobalAlgo,
};
use serde::{Deserialize, Serialize};

/// The scope is the main [operation](Operation) and [variable](Variable) container that simplify
/// the process of reading inputs, creating local variables and adding new operations.
///
/// Notes:
///
/// This type isn't responsible for creating [shader bindings](super::Binding) and figuring out which
/// variable can be written to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scope {
    pub depth: u8,
    pub operations: Vec<Operation>,
    locals: Vec<Variable>,
    reads_global: Vec<(Variable, ReadingStrategy, Variable)>,
    writes_global: Vec<(Variable, Variable)>,
    reads_scalar: Vec<(Variable, Variable)>,
    output_ref: Option<Variable>,
    undeclared: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReadingStrategy {
    /// Each element will be read in a way to be compatible with the output layout.
    OutputLayout,
    /// Keep the current layout.
    Plain,
}

/// Information necessary when compiling a scope.
pub struct ScopeProcessing {
    /// The variable declarations.
    pub variables: Vec<Variable>,
    /// The operations.
    pub operations: Vec<Operation>,
}

impl Scope {
    /// Create a scope that is at the root of a
    /// [compute shader](crate::codegen::dialect::gpu::ComputeShader).
    ///
    /// A local scope can be created with the [child](Self::child) method.
    pub fn root() -> Self {
        Self {
            depth: 0,
            operations: Vec::new(),
            locals: Vec::new(),
            reads_global: Vec::new(),
            writes_global: Vec::new(),
            reads_scalar: Vec::new(),
            output_ref: None,
            undeclared: 0,
        }
    }

    /// Create a local variable of the given [item type](Item).
    pub fn create_local<I: Into<Item>>(&mut self, item: I) -> Variable {
        let item = item.into();
        let index = self.new_local_index();
        let local = Variable::Local(index, item, self.depth);
        self.locals.push(local);
        local
    }

    /// Create a new local variable, but doesn't perform the declaration.
    ///
    /// Useful for _for loops_ and other algorithms that require the control over initialization.
    pub fn create_local_undeclare(&mut self, item: Item) -> Variable {
        let index = self.new_local_index();
        let local = Variable::Local(index, item, self.depth);
        self.undeclared += 1;
        local
    }

    /// Reads an input array to a local variable.
    ///
    /// The index refers to the argument position of the array in the compute shader.
    pub fn read_array<I: Into<Item>>(&mut self, index: u16, item: I) -> Variable {
        self.read_input_strategy(index, item.into(), ReadingStrategy::OutputLayout)
    }

    /// Reads an input scalar to a local variable.
    ///
    /// The index refers to the scalar position for the same [element](Elem) type.
    pub fn read_scalar(&mut self, index: u16, elem: Elem) -> Variable {
        let local = Variable::LocalScalar(self.new_local_scalar_index(), elem, self.depth);
        let scalar = Variable::GlobalScalar(index, elem);

        self.reads_scalar.push((local, scalar));

        local
    }

    /// Retrieve the last local variable that was created.
    pub fn last_local_index(&self) -> Option<&Variable> {
        self.locals.last()
    }

    /// Vectorize the scope using the [vectorization](Vectorization) type.
    ///
    /// Notes:
    ///
    /// Scopes created _during_ compilation (after the tracing is done) should not be vectorized.
    pub fn vectorize(&mut self, vectorization: Vectorization) {
        self.operations
            .iter_mut()
            .for_each(|op| *op = op.vectorize(vectorization));
        self.locals
            .iter_mut()
            .for_each(|var| *var = var.vectorize(vectorization));
        self.reads_global.iter_mut().for_each(|(input, _, output)| {
            *input = input.vectorize(vectorization);
            *output = output.vectorize(vectorization);
        });
        self.writes_global.iter_mut().for_each(|(input, output)| {
            *input = input.vectorize(vectorization);
            *output = output.vectorize(vectorization);
        });
    }

    /// Writes a variable to given output.
    ///
    /// Notes:
    ///
    /// This should only be used when doing compilation.
    pub(crate) fn write_global(&mut self, input: Variable, output: Variable) {
        if self.output_ref.is_none() {
            self.output_ref = Some(output);
        }
        self.writes_global.push((input, output));
    }

    /// Writes a variable to given output.
    ///
    /// Notes:
    ///
    /// This should only be used when doing compilation.
    pub(crate) fn write_global_custom(&mut self, output: Variable) {
        if self.output_ref.is_none() {
            self.output_ref = Some(output);
        }
    }

    /// Update the [reading strategy](ReadingStrategy) for an input array.
    ///
    /// Notes:
    ///
    /// This should only be used when doing compilation.
    pub(crate) fn update_read(&mut self, index: u16, strategy: ReadingStrategy) {
        if let Some((_, strategy_old, _)) = self
            .reads_global
            .iter_mut()
            .find(|(var, _, _)| var.index() == Some(index))
        {
            *strategy_old = strategy;
        }
    }

    /// Register an [operation](Operation) into the scope.
    pub fn register<T: Into<Operation>>(&mut self, operation: T) {
        self.operations.push(operation.into())
    }

    /// Create an empty child scope.
    pub fn child(&mut self) -> Self {
        Self {
            depth: self.depth + 1,
            operations: Vec::new(),
            locals: Vec::new(),
            reads_global: Vec::new(),
            writes_global: Vec::new(),
            reads_scalar: Vec::new(),
            output_ref: None,
            undeclared: 0,
        }
    }

    /// Returns the variables and operations to be declared and executed.
    ///
    /// Notes:
    ///
    /// New operations and variables can be created within the same scope without having name
    /// conflicts.
    pub fn process(&mut self) -> ScopeProcessing {
        self.undeclared += self.locals.len() as u16;

        let mut variables = Vec::new();
        core::mem::swap(&mut self.locals, &mut variables);

        let mut operations = Vec::new();

        for (input, strategy, local) in self.reads_global.drain(..) {
            match strategy {
                ReadingStrategy::OutputLayout => {
                    let output = self.output_ref.expect(
                        "Output should be set when processing an input with output layout.",
                    );
                    operations.push(Operation::Algorithm(Algorithm::ReadGlobalWithLayout(
                        ReadGlobalWithLayoutAlgo {
                            globals: vec![input],
                            layout: output,
                            outs: vec![local],
                        },
                    )));
                }
                ReadingStrategy::Plain => operations.push(Operation::Algorithm(
                    Algorithm::ReadGlobal(ReadGlobalAlgo {
                        global: input,
                        out: local,
                    }),
                )),
            }
        }

        for (local, scalar) in self.reads_scalar.drain(..) {
            operations.push(
                Operator::AssignLocal(UnaryOperator {
                    input: scalar,
                    out: local,
                })
                .into(),
            );
            variables.push(local);
        }

        for op in self.operations.drain(..) {
            operations.push(op);
        }

        for (input, global) in self.writes_global.drain(..) {
            operations.push(Operation::Algorithm(Algorithm::WriteGlobal(
                WriteGlobalAlgo { input, global },
            )))
        }

        ScopeProcessing {
            variables,
            operations,
        }
        .optimize()
    }

    fn new_local_index(&self) -> u16 {
        self.locals.len() as u16 + self.undeclared
    }

    fn new_local_scalar_index(&self) -> u16 {
        self.reads_scalar.len() as u16
    }

    fn read_input_strategy(
        &mut self,
        index: u16,
        item: Item,
        strategy: ReadingStrategy,
    ) -> Variable {
        let input = Variable::GlobalInputArray(index, item);
        let index = self.new_local_index();
        let local = Variable::Local(index, item, self.depth);
        self.reads_global.push((input, strategy, local));
        self.locals.push(local);
        local
    }
}
