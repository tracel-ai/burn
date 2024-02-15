use super::{
    Algorithm, Elem, Item, Operation, Operator, ReadGlobalAlgo, ReadGlobalWithLayoutAlgo,
    UnaryOperator, Variable, Vectorization,
};
use serde::{Deserialize, Serialize};

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

pub struct ScopeProcessing {
    pub variables: Vec<Variable>,
    pub operations: Vec<Operation>,
}

impl Scope {
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

    pub fn create_local<I: Into<Item>>(&mut self, item: I) -> Variable {
        let item = item.into();
        let index = self.new_local_index();
        let local = Variable::Local(index, item, self.depth);
        self.locals.push(local);
        local
    }

    /// Read an input array to a local variable.
    ///
    /// The index refers to the argument position of the array in the compute shader.
    pub fn read_array<I: Into<Item>>(&mut self, index: u16, item: I) -> Variable {
        self.read_input_strategy(index, item.into(), ReadingStrategy::OutputLayout)
    }

    /// Read an input scalar to a local variable.
    ///
    /// The index refers to the scalar position for the same [element](Elem) type.
    pub fn read_scalar(&mut self, index: u16, elem: Elem) -> Variable {
        let local = Variable::LocalScalar(self.new_local_index(), elem.into(), self.depth);
        let scalar = Variable::GlobalScalar(index, elem.into());

        self.reads_scalar.push((local.clone(), scalar));
        self.undeclared += 1;

        local
    }

    pub fn write_global(&mut self, input: Variable, output: Variable) {
        if self.output_ref.is_none() {
            self.output_ref = Some(output);
        }
        self.writes_global.push((input, output));
    }

    pub fn last_local_index(&self) -> u16 {
        self.locals.last().unwrap().index().unwrap()
    }

    pub fn update_read(&mut self, index: u16, strategy: ReadingStrategy) {
        if let Some((_, strategy_old, _)) = self
            .reads_global
            .iter_mut()
            .find(|(var, _, _)| var.index() == Some(index))
        {
            *strategy_old = strategy;
        }
    }

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

    pub fn create_local_undeclare(&mut self, item: Item) -> Variable {
        let index = self.new_local_index();
        let local = Variable::Local(index, item, self.depth);
        self.undeclared += 1;
        local
    }

    pub fn register<T: Into<Operation>>(&mut self, operation: T) {
        self.operations.push(operation.into())
    }

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
                            global: input,
                            layout: output,
                            out: local,
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

        for (input, out) in self.writes_global.drain(..) {
            operations.push(Operation::Operator(Operator::AssignGlobal(UnaryOperator {
                input,
                out,
            })))
        }

        ScopeProcessing {
            variables,
            operations,
        }
    }

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

    fn new_local_index(&self) -> u16 {
        self.locals.len() as u16 + self.undeclared
    }
}
