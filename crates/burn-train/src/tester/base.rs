use crate::checkpoint::{FileCheckpointer, MetricCheckpointingStrategy};
use crate::components::LearnerComponentsMarker;
use crate::components_test::TesterComponents;
use crate::Learner;
use burn_core::optim::adaptor::OptimizerAdaptor;
use burn_core::optim::Adam;
use burn_core::record::{DefaultFileRecorder, FullPrecisionSettings};
use burn_core::tensor::backend::AutodiffBackend;

/// Tester struct which tests a Neural Network model.
///
/// To create a tester, use the [builder](crate::tester::TesterBuilder) struct.
pub struct Tester<TC: TesterComponents> {
    pub(crate) learner: Learner<
        LearnerComponentsMarker<
            TC::Backend,
            f64,
            TC::Model,
            OptimizerAdaptor<
                Adam<<<TC as TesterComponents>::Backend as AutodiffBackend>::InnerBackend>,
                TC::Model,
                TC::Backend,
            >,
            FileCheckpointer<DefaultFileRecorder<FullPrecisionSettings>>,
            FileCheckpointer<DefaultFileRecorder<FullPrecisionSettings>>,
            FileCheckpointer<DefaultFileRecorder<FullPrecisionSettings>>,
            TC::EventProcessor,
            MetricCheckpointingStrategy,
        >,
    >,
}
