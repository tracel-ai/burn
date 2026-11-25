//! The floors of a run, taken in a run of their own so the one timed evaluates none.

use std::cell::RefCell;
use std::rc::Rc;

use crate::{Bounds, ComponentObserver, Declaration};

/// What one entry of a pricing run declared.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Priced {
    /// See [`Declaration::bounds`].
    pub bounds: Option<Bounds>,
    /// See [`Declaration::lead_bounds`].
    pub lead_bounds: Option<Bounds>,
}

/// The floors each entry of a pricing run declared, in the order entered.
///
/// The timed run replays the same closure, so its entries arrive in the same order and take
/// their floors by position. A bounds function evaluated here may read values back: nothing
/// here is timed, so a sync costs the measurement nothing.
#[derive(Default)]
pub struct Floors(std::vec::IntoIter<Priced>);

impl Floors {
    /// The floors of the next entry, or none past the last one priced.
    pub fn next(&mut self) -> Priced {
        self.0.next().unwrap_or_default()
    }
}

/// Records the floors of every entry of a pricing run.
#[derive(Clone)]
pub struct Pricing {
    /// Each entry's floors, in the order entered.
    floors: Rc<RefCell<Vec<Priced>>>,
    /// Whether the run is on a sharded compute device, so its entries price their lead too.
    sharded: bool,
}

impl Pricing {
    /// Records a pricing run, whose entries price their lead too where it is `sharded`.
    pub fn new(sharded: bool) -> Self {
        Self {
            floors: Rc::default(),
            sharded,
        }
    }

    /// The floors recorded so far.
    pub fn floors(&self) -> Floors {
        Floors(self.floors.take().into_iter())
    }
}

impl ComponentObserver for Pricing {
    fn enter(&self, declared: Declaration<'_>) {
        self.floors.borrow_mut().push(Priced {
            bounds: declared.bounds,
            lead_bounds: declared.lead_bounds,
        });
    }

    fn exit(&self) {}

    fn prices_lead(&self) -> bool {
        self.sharded
    }
}
