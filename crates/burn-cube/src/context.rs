use crate::ExpandElement;
use burn_jit::gpu::{Item, Scope, Variable};
use std::{collections::HashMap, sync::Arc};

#[derive(Default, Clone)]
pub struct VariablePool {
    map: std::rc::Rc<std::cell::RefCell<HashMap<Item, Vec<Arc<Variable>>>>>,
}

impl VariablePool {
    pub fn reuse(&self, item: Item) -> Option<ExpandElement> {
        let map = self.map.borrow();

        let variables = match map.get(&item) {
            Some(val) => val,
            None => return None,
        };

        for variable in variables.iter() {
            if Arc::strong_count(variable) == 1 {
                return Some(Arc::clone(variable));
            }
        }

        None
    }
    pub fn insert(&mut self, var: ExpandElement) {
        let mut map = self.map.borrow_mut();
        let item = var.item();

        if let Some(variables) = map.get_mut(&item) {
            variables.push(var.clone());
        } else {
            map.insert(var.item(), vec![var.clone()]);
        }
    }
}

pub struct CubeContext {
    pub scope: Scope,
    pub pool: VariablePool,
}

impl CubeContext {
    pub fn root() -> CubeContext {
        Self {
            pool: Default::default(),
            scope: Scope::root(),
        }
    }
    pub fn child(&mut self) -> CubeContext {
        let scope = self.scope.child();

        Self {
            scope,
            pool: self.pool.clone(),
        }
    }

    pub fn create_local(&mut self, item: Item) -> ExpandElement {
        if let Some(var) = self.pool.reuse(item) {
            return var;
        }

        let new = Arc::new(self.scope.create_local(item));
        self.pool.insert(new.clone());

        new
    }
}
