use burn_core as burn;

use burn::config::Config;
use burn_core::module::ParamGroup;

use crate::lr_scheduler::composed::{LrSchedulerConfig, LrSchedulerItem};

#[derive(Config, Debug)]
struct LrSchedulerGroupConfig {
    group: ParamGroup,
    scheduler: LrSchedulerConfig,
}

struct LrSchedulerGroup {
    group: ParamGroup,
    scheduler: LrSchedulerItem,
}

#[derive(Config, Debug)]
pub struct MultiLrSchedulerConfig {
    default: LrSchedulerConfig,
    scheduler_groups: Vec<LrSchedulerGroupConfig>,
}

pub struct MultiLrScheduler {
    default: LrSchedulerItem,
    scheduler_groups: Vec<LrSchedulerGroup>,
}
