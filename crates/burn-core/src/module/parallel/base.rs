use super::DistributedLayer;
use crate::module::Module;

/// What the hidden layers of a [`LayerParallelism`] model take and return: whatever its input
/// layer produces.
pub type HiddenLayerSignal<M> = <<M as LayerParallelism>::InputLayer as DistributedLayer>::Output;

/// A model that runs as an input layer, then hidden layers in order, then an output layer, so its
/// layers can be split across devices by a [`DistributedLayeredModel`](super::DistributedLayeredModel).
///
/// It is usually not the struct the model trained with, since a layer split across devices may be
/// represented differently. Weights trained on another struct load into this one by remapping
/// their keys. Each layer is built on the device a [`LayerPlacement`](super::LayerPlacement) gives
/// it, so the model never exists whole on one device.
///
/// ```rust,ignore
/// impl Model {
///     fn new(config: &ModelConfig, placement: &LayerPlacement) -> Self {
///         Self {
///             embedding: Embedding::new(config, &placement.input),
///             blocks: placement
///                 .hidden
///                 .iter()
///                 .map(|device| Block::new(config, device))
///                 .collect(),
///             head: Head::new(config, &placement.output),
///         }
///     }
/// }
///
/// impl LayerParallelism for Model {
///     type InputLayer = Embedding;
///     type HiddenLayer = Block;
///     type OutputLayer = Head;
///
///     fn layer_input(&self) -> &Embedding {
///         &self.embedding
///     }
///
///     fn layer_hidden(&self, index: usize) -> Option<&Block> {
///         self.blocks.get(index)
///     }
///
///     fn layer_output(&self) -> &Head {
///         &self.head
///     }
/// }
///
/// let placement = LayerPlacement::even(&devices, config.num_blocks);
/// let model = DistributedLayeredModel::new(Model::new(&config, &placement), &placement);
/// let predictions = model.forward(features);
/// ```
pub trait LayerParallelism: Module {
    /// Runs first, on what the model takes; what it returns is what the hidden layers pass along.
    type InputLayer: DistributedLayer<Input: Module, Output: Module>;
    /// Each hidden layer.
    type HiddenLayer: DistributedLayer<Input = HiddenLayerSignal<Self>, Output = HiddenLayerSignal<Self>>;
    /// Runs last; what it returns is what the model returns.
    type OutputLayer: DistributedLayer<Input = HiddenLayerSignal<Self>>;

    /// The input layer.
    fn layer_input(&self) -> &Self::InputLayer;

    /// The hidden layer at `index` in forward order, `None` past the last one.
    fn layer_hidden(&self, index: usize) -> Option<&Self::HiddenLayer>;

    /// The output layer.
    fn layer_output(&self) -> &Self::OutputLayer;
}
