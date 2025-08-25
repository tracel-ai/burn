use super::{LrScheduler, String};
use crate as burn;
use crate::lr_scheduler::cosine::{CosineAnnealingLrScheduler, CosineAnnealingLrSchedulerConfig};
use crate::lr_scheduler::linear::{LinearLrScheduler, LinearLrSchedulerConfig};
use crate::{LearningRate, config::Config};
use burn_derive::Record;
use burn_tensor::backend::Backend;

#[derive(Config)]
pub struct ComposedLrSchedulerConfig {
    #[config(default = "Vec::new()")]
    schedulers: Vec<LrSchedulerConfig>,
    #[config(default = "SchedulerReduction::Prod")]
    reduction: SchedulerReduction,
}

#[derive(Config, Copy)]
pub enum SchedulerReduction {
    Avg,
    Prod,
}

impl ComposedLrSchedulerConfig {
    pub fn init(&self) -> Result<ComposedLrScheduler, String> {
        let mut schedulers = Vec::with_capacity(self.schedulers.len());
        for config in self.schedulers.iter() {
            let config = match config {
                LrSchedulerConfig::Linear(config) => LrSchedulerItem::Linear(config.init()?),
                LrSchedulerConfig::Cosine(config) => LrSchedulerItem::Cosine(config.init()?),
            };
            schedulers.push(config);
        }

        Ok(ComposedLrScheduler {
            schedulers,
            reduction: self.reduction,
        })
    }

    pub fn linear(mut self, config: LinearLrSchedulerConfig) -> Self {
        self.schedulers.push(LrSchedulerConfig::Linear(config));
        self
    }

    pub fn consine(mut self, config: CosineAnnealingLrSchedulerConfig) -> Self {
        self.schedulers.push(LrSchedulerConfig::Cosine(config));
        self
    }
}

#[derive(Config)]
pub enum LrSchedulerConfig {
    Linear(LinearLrSchedulerConfig),
    Cosine(CosineAnnealingLrSchedulerConfig),
}

#[derive(Clone)]
pub enum LrSchedulerItem {
    Linear(LinearLrScheduler),
    Cosine(CosineAnnealingLrScheduler),
}

#[derive(Record)]
pub enum LrSchedulerRecord<B: Backend> {
    Linear(<LinearLrScheduler as LrScheduler>::Record<B>),
    Cosine(<CosineAnnealingLrScheduler as LrScheduler>::Record<B>),
}

#[derive(Clone)]
pub struct ComposedLrScheduler {
    schedulers: Vec<LrSchedulerItem>,
    reduction: SchedulerReduction,
}

#[derive(Record)]
pub struct ComposedLrSchedulerRecord<B: Backend> {
    schedulers: Vec<LrSchedulerRecord<B>>,
}

impl LrScheduler for ComposedLrScheduler {
    type Record<B: Backend> = ComposedLrSchedulerRecord<B>;

    fn step(&mut self) -> LearningRate {
        let mut step = match self.reduction {
            SchedulerReduction::Avg => 0.0,
            SchedulerReduction::Prod => 1.0,
        };
        let num_scheduler = self.schedulers.len() as f64;
        for lr in self.schedulers.iter_mut().map(|s| match s {
            LrSchedulerItem::Linear(item) => item.step(),
            LrSchedulerItem::Cosine(item) => item.step(),
        }) {
            step = match self.reduction {
                SchedulerReduction::Avg => step + (lr / num_scheduler),
                SchedulerReduction::Prod => step * lr,
            }
        }

        step
    }

    fn to_record<B: Backend>(&self) -> Self::Record<B> {
        ComposedLrSchedulerRecord::<B> {
            schedulers: self
                .schedulers
                .iter()
                .map(|s| match s {
                    LrSchedulerItem::Linear(item) => {
                        LrSchedulerRecord::Linear(item.to_record::<B>())
                    }
                    LrSchedulerItem::Cosine(item) => {
                        LrSchedulerRecord::Linear(item.to_record::<B>())
                    }
                })
                .collect(),
        }
    }

    fn load_record<B: Backend>(mut self, record: Self::Record<B>) -> Self {
        self.schedulers = self
            .schedulers
            .into_iter()
            .zip(record.schedulers.into_iter())
            .map(|scheduler| match scheduler {
                (LrSchedulerItem::Linear(item), LrSchedulerRecord::Linear(record)) => {
                    LrSchedulerItem::Linear(item.load_record::<B>(record))
                }
                (LrSchedulerItem::Cosine(item), LrSchedulerRecord::Cosine(record)) => {
                    LrSchedulerItem::Cosine(item.load_record::<B>(record))
                }
                _ => panic!("Invalid state"),
            })
            .collect();

        self
    }
}
