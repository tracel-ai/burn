use crate::ExpandElement;
use alloc::rc::Rc;
use burn_jit::gpu::{self, Item, Scope};
use core::cell::RefCell;
use std::collections::HashMap;

#[derive(Default, Clone)]
pub struct VariablePool {
    map: Rc<RefCell<HashMap<Item, Vec<ExpandElement>>>>,
}

impl VariablePool {
    /// Returns an old, not used anymore variable, if there exists one.
    pub fn reuse(&self, item: Item) -> Option<ExpandElement> {
        let map = self.map.borrow();

        // Filter for candidate variables of the same Item
        let variables = match map.get(&item) {
            Some(val) => val,
            None => return None,
        };

        // Among the candidates, take a variable if it's only referenced by the map
        // Arbitrarily takes the first it finds
        for variable in variables.iter() {
            if Rc::strong_count(&variable.inner) == 1 {
                // println!("Reuse var {:?}", variable.inner);
                return Some(variable.clone());
            }
        }

        // If no candidate was found, a new var will be needed
        None
    }

    /// Insert a new variable in the map, which is classified by Item
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
    pub root: Rc<RefCell<Scope>>,
    pub scope: Rc<RefCell<Scope>>,
    pub pool: VariablePool,
}

impl CubeContext {
    /// Create a new cube context, with a root scope
    /// A root scope is at the root of a compute shader
    /// Therefore there is one cube context per shader
    pub fn root() -> CubeContext {
        let root = Rc::new(RefCell::new(Scope::root()));
        let scope = root.clone();

        Self {
            pool: Default::default(),
            scope,
            root,
        }
    }

    pub fn register<O: Into<gpu::Operation>>(&mut self, op: O) {
        self.scope.borrow_mut().register(op)
    }

    pub fn child(&mut self) -> CubeContext {
        let scope = self.scope.borrow_mut().child();

        Self {
            scope: Rc::new(RefCell::new(scope)),
            root: self.root.clone(),
            pool: self.pool.clone(),
        }
    }

    pub fn into_scope(self) -> Scope {
        core::mem::drop(self.root);

        Rc::into_inner(self.scope)
            .expect("Only one reference")
            .into_inner()
    }

    /// When a new variable is required, we check if we can reuse an old one
    /// Otherwise we create a new one.
    pub fn create_local(&mut self, item: Item) -> ExpandElement {
        // Reuse an old variable if possible
        if let Some(var) = self.pool.reuse(item) {
            return var;
        }

        // Create a new variable at the root scope
        // Insert it in the variable pool for potential reuse
        let new = ExpandElement::new(Rc::new(self.root.borrow_mut().create_local(item)));
        self.pool.insert(new.clone());

        new
    }
}
