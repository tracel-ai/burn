use std::any::Any;

use burn_optim::lr_scheduler::module_lr_scheduler::ModuleLearningRate;
use burn_std::ExecutionError;

use crate::ItemLazy;

/// Observations captured for an applied optimizer update.
///
/// Built-in strategies supply learning rates. Custom strategies can attach additional typed
/// observations with [`with_observation`](Self::with_observation). Each type has at most one
/// value per update; use distinct wrapper types for observations with different meanings.
/// Extensions are synchronized on the reporting thread before any optimizer metrics run.
/// A failure to synchronize any extension rejects the entire update's observations.
pub struct OptimizerStepOutput {
    learning_rate: ModuleLearningRate,
    observations: Vec<Box<dyn ErasedObservation>>,
}

impl OptimizerStepOutput {
    /// Capture the learning rates used for the applied update.
    /// No storage for custom observations is allocated until an observation is attached.
    pub fn new(learning_rate: ModuleLearningRate) -> Self {
        Self {
            learning_rate,
            observations: Vec::new(),
        }
    }

    /// Learning rates used for this update, in scheduler registration order.
    pub fn learning_rate(&self) -> &ModuleLearningRate {
        &self.learning_rate
    }

    /// Attach an observation, replacing any previous observation of the same type.
    ///
    /// The strategy owns collection and its cost. Implement [`ItemLazy`] to flush pending tensor
    /// work before metrics read the observation, and detach tensors from autodiff before attaching
    /// them so the reporting thread does not retain the training graph.
    pub fn with_observation<T: ItemLazy + 'static>(mut self, observation: T) -> Self {
        let value = Box::new(Observation(Some(observation)));
        match self
            .observations
            .iter_mut()
            .find(|entry| entry.as_any().is::<T>())
        {
            Some(entry) => *entry = value,
            None => self.observations.push(value),
        }
        self
    }

    /// Read an observation by type. Returns `None` if this update did not supply it.
    ///
    /// A metric requiring this observation should return an error when it is absent. Metrics
    /// for optional observations can instead return an unavailable entry. Observations are local to
    /// this optimizer step; an absent value never falls back to an earlier step's value.
    pub fn observation<T: 'static>(&self) -> Option<&T> {
        self.observations
            .iter()
            .find_map(|entry| entry.as_any().downcast_ref())
    }
}

impl ItemLazy for OptimizerStepOutput {
    fn sync(mut self) -> Result<Self, ExecutionError> {
        for observation in &mut self.observations {
            observation.sync()?;
        }
        Ok(self)
    }
}

trait ErasedObservation: Send {
    fn as_any(&self) -> &dyn Any;
    fn sync(&mut self) -> Result<(), ExecutionError>;
}

// ItemLazy consumes its value. The slot lets us synchronize in the existing allocation; a failed
// sync drops the entire output, so no metric can read an empty slot.
struct Observation<T>(Option<T>);

impl<T: ItemLazy + 'static> ErasedObservation for Observation<T> {
    fn as_any(&self) -> &dyn Any {
        self.0
            .as_ref()
            .expect("Observation is present before and after a successful sync")
    }

    fn sync(&mut self) -> Result<(), ExecutionError> {
        let observation = self.0.take().expect("Observation is present before sync");
        self.0 = Some(observation.sync()?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Pending(u32);
    impl ItemLazy for Pending {
        fn sync(self) -> Result<Self, ExecutionError> {
            Ok(Self(self.0 + 1))
        }
    }
    struct Label(&'static str);
    impl ItemLazy for Label {
        fn sync(self) -> Result<Self, ExecutionError> {
            Ok(self)
        }
    }

    #[test]
    fn typed_observations_replace_by_type_and_sync_the_retained_values() {
        let output = OptimizerStepOutput::new(0.1.into())
            .with_observation(Pending(1))
            .with_observation(Label("module"))
            .with_observation(Pending(10))
            .sync()
            .unwrap();
        assert_eq!(output.observation::<Pending>().unwrap().0, 11);
        assert_eq!(output.observation::<Label>().unwrap().0, "module");
        assert!(output.observation::<String>().is_none());
        assert!(
            OptimizerStepOutput::new(0.1.into())
                .observation::<Pending>()
                .is_none()
        );
    }
}
