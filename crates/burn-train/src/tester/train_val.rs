use crate::components_test::TesterComponents;
use crate::metric::processor::EventProcessor;
use crate::{Tester, TrainStep, ValidStep};
use burn_core::data::dataloader::DataLoader;
use burn_core::module::AutodiffModule;
use std::sync::Arc;

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
        dataloader_valid: Arc<dyn DataLoader<InputValid>>,
    ) -> LC::Model
    where
        InputTrain: Send + 'static,
        InputValid: Send,
        OutputTrain: Send + 'static,
        OutputValid: Send,
        LC::Model: TrainStep<InputTrain, OutputTrain>,
        <LC::Model as AutodiffModule<LC::Backend>>::InnerModule: ValidStep<InputValid, OutputValid>,
        LC::EventProcessor: EventProcessor<ItemTrain = OutputTrain, ItemValid = OutputValid>,
    {
        self.learner.fit(dataloader_train, dataloader_valid)
    }
}
