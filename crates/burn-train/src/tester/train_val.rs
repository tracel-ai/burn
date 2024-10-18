use crate::components_test::TesterComponents;
use crate::metric::processor::EventProcessor;
use crate::{Tester, TrainStep, ValidStep};
use burn_core::data::dataloader::DataLoader;
use burn_core::module::AutodiffModule;
use std::sync::Arc;

use super::empty_data_loader::EmptyDataLoader;

/// A training output.
impl<LC: TesterComponents> Tester<LC> {
    /// Fits the model.
    ///
    /// # Arguments
    ///
    /// * `dataloader_train` - The training dataloader.
    /// * `dataloader_valid` - The validation dataloader.
    ///
    /// # Returns
    ///
    /// The fitted model.
    pub fn test<InputTrain, InputValid, OutputTrain, OutputValid>(
        self,
        dataloader_train: Arc<dyn DataLoader<InputTrain>>,
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
        self.learner
            .fit(dataloader_train, Arc::new(EmptyDataLoader {}))
    }
}
