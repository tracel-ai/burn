use super::{Item, Operation, Variable};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scope {
    pub depth: u8,
    pub operations: Vec<Operation>,
    pub num_local_variables: u16,
}

impl Scope {
    pub fn root() -> Self {
        Self {
            depth: 0,
            operations: Vec::new(),
            num_local_variables: 0,
        }
    }

    pub fn create_local(&mut self, item: Item) -> Variable {
        let index = self.num_local_variables;
        self.num_local_variables += 1;
        Variable::Local(index, item, self.depth)
    }

    pub fn register<T: Into<Operation>>(&mut self, operation: T) {
        self.operations.push(operation.into())
    }

    pub fn child(&mut self) -> Self {
        Self {
            depth: self.depth + 1,
            operations: Vec::new(),
            num_local_variables: 0,
        }
    }
}
