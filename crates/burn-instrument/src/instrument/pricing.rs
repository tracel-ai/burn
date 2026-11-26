//! The floors of a run, taken in a run of their own so the one timed evaluates none.

use std::cell::RefCell;
use std::rc::Rc;

use crate::{Bounds, ComponentObserver, Declaration};

/// What one entry of the pricing runs declared.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Priced {
    /// See [`Declaration::bounds`].
    pub bounds: Option<Bounds>,
    /// The same declaration priced at the lead's share, by a pricing run of the lead (see
    /// [`pricing_lead`](crate::pricing_lead)).
    pub lead_bounds: Option<Bounds>,
}

/// The floors each entry of the pricing runs declared, in the order entered.
///
/// The timed run replays the same closure, so its entries arrive in the same order and take
/// their floors by position. A bounds function evaluated here may read values back: nothing
/// here is timed, so a sync costs the measurement nothing.
#[derive(Default)]
pub struct Floors {
    /// Each entry's floor for the device.
    device: std::vec::IntoIter<Option<Bounds>>,
    /// Each entry's floor at the lead's share, empty where no run priced the lead.
    lead: std::vec::IntoIter<Option<Bounds>>,
}

impl Floors {
    /// The floors of a run priced for the device, and of the same run priced for its lead.
    pub fn new(device: Pricing, lead: Option<Pricing>) -> Self {
        Self {
            device: device.floors.take().into_iter(),
            lead: lead
                .map(|lead| lead.floors.take())
                .unwrap_or_default()
                .into_iter(),
        }
    }

    /// The floors of the next entry, or none past the last one priced.
    pub fn next(&mut self) -> Priced {
        Priced {
            bounds: self.device.next().flatten(),
            lead_bounds: self.lead.next().flatten(),
        }
    }
}

/// Records the floor of every entry of a pricing run.
#[derive(Clone)]
pub struct Pricing {
    /// Each entry's floor, in the order entered.
    floors: Rc<RefCell<Vec<Option<Bounds>>>>,
    /// Whether the entries price their bounds at the lead's share.
    lead: bool,
}

impl Pricing {
    /// Records a pricing run, of the lead's share where `lead`.
    pub fn new(lead: bool) -> Self {
        Self {
            floors: Rc::default(),
            lead,
        }
    }
}

impl ComponentObserver for Pricing {
    fn enter(&self, declared: Declaration<'_>) {
        self.floors.borrow_mut().push(declared.bounds);
    }

    fn exit(&self) {}

    fn prices_lead(&self) -> bool {
        self.lead
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An entry of `pricing` that declared `read` bytes.
    fn declare(pricing: &Pricing, read: u64) {
        pricing.enter(Declaration {
            label: "attention",
            variant: None,
            site: "test::forward",
            bounds: Some(Bounds {
                read,
                write: 0,
                flops: 0,
            }),
        });
    }

    #[test]
    fn an_entry_takes_its_floors_from_the_device_and_the_leads_runs() {
        let (device, lead) = (Pricing::new(false), Pricing::new(true));
        declare(&device, 4);
        declare(&lead, 2);

        let floors = Floors::new(device, Some(lead)).next();

        assert_eq!(floors.bounds.map(|bounds| bounds.read), Some(4));
        assert_eq!(floors.lead_bounds.map(|bounds| bounds.read), Some(2));
    }

    #[test]
    fn a_run_not_priced_at_the_lead_declares_no_lead_floor() {
        let device = Pricing::new(false);
        declare(&device, 4);

        let floors = Floors::new(device, None).next();

        assert_eq!(floors.lead_bounds, None);
    }
}
