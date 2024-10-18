use crate::components_test::TesterComponents;
use crate::metric::processor::EventProcessor;
use crate::{Tester, TrainStep, ValidStep};
use burn_core::data::dataloader::DataLoader;
use burn_core::module::AutodiffModule;
use std::sync::Arc;

use super::empty_data_loader::EmptyDataLoader;

/// A testing output.
impl<LC: TesterComponents> Tester<LC> {
    /// Tests the model.
    ///
    /// # Arguments
    ///
    /// * `dataloader` - The testing dataloader.
    ///
    /// # Returns
    ///
    /// The tested model.
    pub fn test<InputTrain, InputValid, OutputTrain, OutputValid>(
        self,
        dataloader: Arc<dyn DataLoader<InputTrain>>,
    ) -> LC::Model
    where
        InputTrain: Send + 'static,
        InputValid: Send + 'static,
        OutputTrain: Send + 'static,
        OutputValid: Send,
        LC::Model: TrainStep<InputTrain, OutputTrain>,
        <LC::Model as AutodiffModule<LC::Backend>>::InnerModule: ValidStep<InputValid, OutputValid>,
        LC::EventProcessor: EventProcessor<ItemTrain = OutputTrain, ItemValid = OutputValid>,
    {
        self.learner.fit(dataloader, Arc::new(EmptyDataLoader {}))
    }
}
