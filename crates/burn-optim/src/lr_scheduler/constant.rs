use super::{LrScheduler, LrSchedulerRecord};
use crate::HostLr;

/// Constant learning rate implementing [learning rate scheduler](LrScheduler).
///
/// # Notes
///
/// You can also use [learning rate](HostLr) with the same effect.
#[derive(new, Clone, Debug)]
pub struct ConstantLr {
    lr: HostLr,
}

impl From<HostLr> for ConstantLr {
    fn from(lr: HostLr) -> Self {
        Self { lr }
    }
}

impl LrScheduler for ConstantLr {
    fn step(&mut self) -> HostLr {
        self.lr
    }

    fn to_record(&self) -> LrSchedulerRecord {
        LrSchedulerRecord::new()
    }

    fn load_record(&mut self, _record: LrSchedulerRecord) {}
}

impl LrScheduler for HostLr {
    fn step(&mut self) -> HostLr {
        *self
    }

    fn to_record(&self) -> LrSchedulerRecord {
        LrSchedulerRecord::new()
    }

    fn load_record(&mut self, _record: LrSchedulerRecord) {}
}
