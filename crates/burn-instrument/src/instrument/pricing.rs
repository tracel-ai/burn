//! The floors of a run, taken in a run of their own so the one timed evaluates none.

use std::cell::RefCell;
use std::rc::Rc;

use crate::{Bounds, ComponentObserver, Declaration};

/// The floor each entry of a pricing run declared, in the order entered.
///
/// The timed run replays the same closure, so its entries arrive in the same order and take
/// their floors by position. A bounds function evaluated here may read values back: nothing
/// here is timed, so a sync costs the measurement nothing.
#[derive(Default)]
pub struct Floors(std::vec::IntoIter<Option<Bounds>>);

impl Floors {
    /// The floor of the next entry, or `None` past the last one priced.
    pub fn next(&mut self) -> Option<Bounds> {
        self.0.next().flatten()
    }
}

/// Records the floor of every entry of a pricing run.
#[derive(Clone, Default)]
pub struct Pricing(Rc<RefCell<Vec<Option<Bounds>>>>);

impl Pricing {
    /// The floors recorded so far.
    pub fn floors(&self) -> Floors {
        Floors(self.0.take().into_iter())
    }
}

impl ComponentObserver for Pricing {
    fn enter(&self, declared: Declaration<'_>) {
        self.0.borrow_mut().push(declared.bounds);
    }

    fn exit(&self) {}
}
