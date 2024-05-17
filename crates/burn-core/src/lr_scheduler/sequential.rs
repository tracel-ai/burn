use super::LrScheduler;
use crate as burn;
use crate::{config::Config, LearningRate};
use burn_tensor::backend::Backend;

/// The configuration for creating a sequential LR scheduler from a sequence of LR schedulers.
///
/// Each scheduler is run until the specified start step of the next scheduler. The last scheduler
/// will be run until the end of training
#[derive(Config)]
pub struct SequentialLrSchedulerConfig {
    // The configs for each of the each of the component schedulers
    component_lr_configs: Vec<Config>,

    // The start steps for each of the component schedulers. If there
    // are n components, there should be n-1 start steps, as the first
    // component is started at the first step
    component_lr_start_steps: Vec<i64>
}

impl SequentialLrSchedulerConfig {
    /// Initializes a [exponential learning rate scheduler](ExponentialLrScheduler).
    ///
    /// # Panics
    /// This function panics if `initial_lr` and `gamma` are not between 0 and 1.
    pub fn init(&self) -> ExponentialLrScheduler {
        assert!(
            self.component_lr_configs.len() == self.component_lr_start_steps.len() + 1,
            "Should have one less component_lr_start_steps than component_lr_configs"
        );

        // build the schedulers
        let mut component_lr_schedulers = Vec::new();

        for clc in self.component_lr_configs {
            component_lr_schedulers.push(
                clc.init()
            );
        }

        let first = component_lr_schedulers.pop();

        SequentialLrScheduler {
            current_lr_schedulr: first,
            remaining_lr_schedulers: self.component_lr_configs,
            next_steps: self.component_lr_start_steps,
            curr_step: 0,
        }
    }
}

/// A exponential learning rate scheduler.
///
/// See [ExponentialLrSchedulerConfig] for more information.
#[derive(Clone, Copy, Debug)]
pub struct SequentialLrScheduler {
    current_lr_schedulr: LrScheduler,
    remaining_lr_schedulers: Vec<LrScheduler>,
    next_steps: Vec<i64>,
    curr_step: i64,
}

impl<B: Backend> LrScheduler<B> for ExponentialLrScheduler {
    type Record = (LearningRate, f64);

    fn step(&mut self) -> LearningRate {

        if let ns = Some(self.next_steps.pop()) & self.curr_step > ns{
            // select the next scheduler
            self.current_lr_schedulr = self.remaining_lr_schedulers.pop();
        }

        self.curr_step += 1;

        self.current_lr_schedulr.step()
    }

    fn to_record(&self) -> Self::Record {
        (
            self.current_lr_schedulr.map(|x| x.to_record()).collect(), 
            self.remaining_lr_schedulers.to_record().collect(),
            self.next_steps,
            self.curr_step,
        )
    }

    fn load_record(mut self, record: Self::Record) -> Self {
        //FIXME
        (self.current_lr_schedulr, self.remaining_lr_schedulers, self.next_steps, self.curr_step) = record
        self
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::TestBackend;

    #[test]
    #[should_panic = "Should have one less component_lr_start_steps than component_lr_configs"]
    fn wrong_num_of_components() {
        SequentialLrSchedulerConfig::new(
            vec![
                ExponentialLrSchedulerConfig::new(0.5, 0.1),
                ExponentialLrSchedulerConfig::new(0.5, 0.1)
            ],
            vec![3, 3]
        ).init();
    }

    // #[test]
    // fn test_lr_change() {
    //     const INITIAL_LR: LearningRate = 0.75;
    //     const GAMMA: f64 = 0.9;
    //     const NUM_ITERS: usize = 10;
    //     const EPSILON: f64 = 1e-10;

    //     let mut scheduler = ExponentialLrSchedulerConfig::new(INITIAL_LR, GAMMA).init();

    //     let mut previous_lr = INITIAL_LR;

    //     for _ in 0..NUM_ITERS {
    //         let lr = LrScheduler::<TestBackend>::step(&mut scheduler);
    //         assert!(
    //             lr < previous_lr,
    //             "Learning rate should decrease with each iteration before reaching the final learning rate"
    //         );
    //         previous_lr = lr;
    //     }

    //     let expected = INITIAL_LR * GAMMA.powi(NUM_ITERS as i32);
    //     assert!(
    //         (previous_lr - expected).abs() < EPSILON,
    //         "Learning rate should be close to the expected value after reaching the final learning rate"
    //     );
    // }
}
