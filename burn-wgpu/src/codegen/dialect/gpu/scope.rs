use super::{Item, Operation, Variable};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scope {
    pub prefix: String,
    pub operations: Vec<Operation>,
    pub num_local_variables: u16,
}

impl Scope {
    pub fn new(prefix: String) -> Self {
        Self {
            prefix,
            operations: Vec::new(),
            num_local_variables: 0,
        }
    }

    pub fn create_local(&mut self, item: Item) -> Variable {
        let index = self.num_local_variables;
        self.num_local_variables += 1;
        Variable::Local(index, item)
    }

    pub fn register(&mut self, operation: Operation) {
        self.operations.push(operation)
    }
}
