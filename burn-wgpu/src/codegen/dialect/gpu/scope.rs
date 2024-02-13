use super::{Item, Operation, Variable};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scope {
    pub depth: u8,
    pub operations: Vec<Operation>,
    pub locals: Vec<Variable>,
    pub undeclared: u16,
}

impl Scope {
    pub fn root() -> Self {
        Self {
            depth: 0,
            operations: Vec::new(),
            locals: Vec::new(),
            undeclared: 0,
        }
    }

    pub fn create_local(&mut self, item: Item) -> Variable {
        let index = self.locals.len() as u16 + self.undeclared;
        let local = Variable::Local(index, item, self.depth);
        self.locals.push(local.clone());
        local
    }

    pub fn create_local_undeclare(&mut self, item: Item) -> Variable {
        let index = self.locals.len() as u16 + self.undeclared;
        let local = Variable::Local(index, item, self.depth);
        self.undeclared += 1;
        local
    }

    pub fn register<T: Into<Operation>>(&mut self, operation: T) {
        self.operations.push(operation.into())
    }

    pub fn process(&mut self) -> (Vec<Operation>, Vec<Variable>) {
        self.undeclared += self.locals.len() as u16;

        let mut operations = Vec::new();
        let mut locals = Vec::new();

        core::mem::swap(&mut self.operations, &mut operations);
        core::mem::swap(&mut self.locals, &mut locals);

        (operations, locals)
    }

    pub fn fork(&self) -> Self {
        Self {
            depth: self.depth,
            operations: Vec::new(), // TODO: Fix Just a hack to not keep the operations.
            locals: self.locals.clone(),
            undeclared: self.undeclared + self.locals.len() as u16,
        }
    }

    pub fn child(&mut self) -> Self {
        Self {
            depth: self.depth + 1,
            operations: Vec::new(),
            locals: Vec::new(),
            undeclared: 0,
        }
    }
}
