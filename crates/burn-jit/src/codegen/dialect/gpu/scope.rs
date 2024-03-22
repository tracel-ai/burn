use crate::JitElement;

use super::{
    gpu, processing::ScopeProcessing, Elem, IndexOffsetGlobalWithLayout, Item, Operation, Operator,
    Procedure, ReadGlobal, ReadGlobalWithLayout, UnaryOperator, Variable, Vectorization,
    WriteGlobal,
};
use serde::{Deserialize, Serialize};

/// The scope is the main [operation](Operation) and [variable](Variable) container that simplify
/// the process of reading inputs, creating local variables and adding new operations.
///
/// Notes:
///
/// This type isn't responsible for creating [shader bindings](super::Binding) and figuring out which
/// variable can be written to.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub struct Scope {
    pub depth: u8,
    pub operations: Vec<Operation>,
    locals: Vec<Variable>,
    shared_memories: Vec<Variable>,
    local_arrays: Vec<Variable>,
    reads_global: Vec<(Variable, ReadingStrategy, Variable)>,
    index_offset_with_output_layout_position: Vec<usize>,
    writes_global: Vec<(Variable, Variable)>,
    reads_scalar: Vec<(Variable, Variable)>,
    pub layout_ref: Option<Variable>,
    undeclared: u16,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub enum ReadingStrategy {
    /// Each element will be read in a way to be compatible with the output layout.
    OutputLayout,
    /// Keep the current layout.
    Plain,
}

impl Scope {
    /// Create a scope that is at the root of a
    /// [compute shader](crate::codegen::dialect::gpu::ComputeShader).
    ///
    /// A local scope can be created with the [child](Self::child) method.
    pub(crate) fn root() -> Self {
        Self {
            depth: 0,
            operations: Vec::new(),
            locals: Vec::new(),
            local_arrays: Vec::new(),
            shared_memories: Vec::new(),
            reads_global: Vec::new(),
            index_offset_with_output_layout_position: Vec::new(),
            writes_global: Vec::new(),
            reads_scalar: Vec::new(),
            layout_ref: None,
            undeclared: 0,
        }
    }

    /// Create a variable initialized at zero.
    pub(crate) fn zero<I: Into<Item>>(&mut self, item: I) -> Variable {
        let local = self.create_local(item);
        let zero: Variable = 0u32.into();
        gpu!(self, local = zero);
        local
    }

    /// Create a variable initialized at some value.
    pub(crate) fn create_with_value<E: JitElement, I: Into<Item> + Copy>(
        &mut self,
        value: E,
        item: I,
    ) -> Variable {
        let local = self.create_local(item);
        let value = Variable::ConstantScalar(value.to_f64().unwrap(), item.into().elem());
        gpu!(self, local = value);
        local
    }

    /// Create a local variable of the given [item type](Item).
    pub(crate) fn create_local<I: Into<Item>>(&mut self, item: I) -> Variable {
        let item = item.into();
        let index = self.new_local_index();
        let local = Variable::Local(index, item, self.depth);
        self.locals.push(local);
        local
    }

    /// Create a new local variable, but doesn't perform the declaration.
    ///
    /// Useful for _for loops_ and other algorithms that require the control over initialization.
    pub(crate) fn create_local_undeclared(&mut self, item: Item) -> Variable {
        let index = self.new_local_index();
        let local = Variable::Local(index, item, self.depth);
        self.undeclared += 1;
        local
    }

    /// Reads an input array to a local variable.
    ///
    /// The index refers to the argument position of the array in the compute shader.
    pub(crate) fn read_array<I: Into<Item>>(&mut self, index: u16, item: I) -> Variable {
        self.read_input_strategy(index, item.into(), ReadingStrategy::OutputLayout)
    }

    /// Add the procedure into the scope.
    pub(crate) fn index_offset_with_output_layout(&mut self, proc: IndexOffsetGlobalWithLayout) {
        self.index_offset_with_output_layout_position
            .push(self.operations.len());
        self.operations
            .push(Procedure::IndexOffsetGlobalWithLayout(proc).into());
    }

    /// Reads an input scalar to a local variable.
    ///
    /// The index refers to the scalar position for the same [element](Elem) type.
    pub(crate) fn read_scalar(&mut self, index: u16, elem: Elem) -> Variable {
        let local = Variable::LocalScalar(self.new_local_scalar_index(), elem, self.depth);
        let scalar = Variable::GlobalScalar(index, elem);

        self.reads_scalar.push((local, scalar));

        local
    }

    /// Retrieve the last local variable that was created.
    pub(crate) fn last_local_index(&self) -> Option<&Variable> {
        self.locals.last()
    }

    /// Vectorize the scope using the [vectorization](Vectorization) type.
    ///
    /// Notes:
    ///
    /// Scopes created _during_ compilation (after the tracing is done) should not be vectorized.
    pub(crate) fn vectorize(&mut self, vectorization: Vectorization) {
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
        // This assumes that all outputs have the same layout
        if self.layout_ref.is_none() {
            self.layout_ref = Some(output);
        }
        self.writes_global.push((input, output));
    }

    /// Writes a variable to given output.
    ///
    /// Notes:
    ///
    /// This should only be used when doing compilation.
    pub(crate) fn write_global_custom(&mut self, output: Variable) {
        // This assumes that all outputs have the same layout
        if self.layout_ref.is_none() {
            self.layout_ref = Some(output);
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

    #[allow(dead_code)]
    pub(crate) fn read_globals(&self) -> Vec<(u16, ReadingStrategy)> {
        self.reads_global
            .iter()
            .map(|(var, strategy, _)| match var {
                Variable::GlobalInputArray(id, _) => (*id, *strategy),
                _ => panic!("Can only read global input arrays."),
            })
            .collect()
    }

    /// Register an [operation](Operation) into the scope.
    pub(crate) fn register<T: Into<Operation>>(&mut self, operation: T) {
        self.operations.push(operation.into())
    }

    /// Create an empty child scope.
    pub(crate) fn child(&mut self) -> Self {
        Self {
            depth: self.depth + 1,
            operations: Vec::new(),
            locals: Vec::new(),
            shared_memories: Vec::new(),
            local_arrays: Vec::new(),
            reads_global: Vec::new(),
            index_offset_with_output_layout_position: Vec::new(),
            writes_global: Vec::new(),
            reads_scalar: Vec::new(),
            layout_ref: self.layout_ref,
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

        for index in self.index_offset_with_output_layout_position.drain(..) {
            if let Some(Operation::Procedure(Procedure::IndexOffsetGlobalWithLayout(proc))) =
                self.operations.get_mut(index)
            {
                proc.layout = self.layout_ref.expect(
                    "Output should be set when processing an index offset with output layout.",
                );
            }
        }

        let mut operations = Vec::new();

        for (input, strategy, local) in self.reads_global.drain(..) {
            match strategy {
                ReadingStrategy::OutputLayout => {
                    let output = self.layout_ref.expect(
                        "Output should be set when processing an input with output layout.",
                    );
                    operations.push(Operation::Procedure(Procedure::ReadGlobalWithLayout(
                        ReadGlobalWithLayout {
                            globals: vec![input],
                            layout: output,
                            outs: vec![local],
                        },
                    )));
                }
                ReadingStrategy::Plain => {
                    operations.push(Operation::Procedure(Procedure::ReadGlobal(ReadGlobal {
                        global: input,
                        out: local,
                    })))
                }
            }
        }

        for (local, scalar) in self.reads_scalar.drain(..) {
            operations.push(
                Operator::Assign(UnaryOperator {
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
            operations.push(Operation::Procedure(Procedure::WriteGlobal(WriteGlobal {
                input,
                global,
            })))
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

    fn new_shared_index(&self) -> u16 {
        self.shared_memories.len() as u16
    }

    fn new_local_array_index(&self) -> u16 {
        self.local_arrays.len() as u16
    }

    fn read_input_strategy(
        &mut self,
        index: u16,
        item: Item,
        strategy: ReadingStrategy,
    ) -> Variable {
        let item_global = match item.elem() {
            Elem::Bool => match item {
                Item::Vec4(_) => Item::Vec4(Elem::UInt),
                Item::Vec3(_) => Item::Vec3(Elem::UInt),
                Item::Vec2(_) => Item::Vec2(Elem::UInt),
                Item::Scalar(_) => Item::Scalar(Elem::UInt),
            },
            _ => item,
        };
        let input = Variable::GlobalInputArray(index, item_global);
        let index = self.new_local_index();
        let local = Variable::Local(index, item, self.depth);
        self.reads_global.push((input, strategy, local));
        self.locals.push(local);
        local
    }

    /// Create a shared variable of the given [item type](Item).
    pub fn create_shared<I: Into<Item>>(&mut self, item: I, shared_memory_size: u32) -> Variable {
        let item = item.into();
        let index = self.new_shared_index();
        let shared_memory = Variable::SharedMemory(index, item, shared_memory_size);
        self.shared_memories.push(shared_memory);
        shared_memory
    }

    /// Create a local array of the given [item type](Item).
    pub fn create_local_array<I: Into<Item>>(&mut self, item: I, array_size: u32) -> Variable {
        let item = item.into();
        let index = self.new_local_array_index();
        let local_array = Variable::LocalArray(index, item, self.depth, array_size);
        self.local_arrays.push(local_array);
        local_array
    }
}
