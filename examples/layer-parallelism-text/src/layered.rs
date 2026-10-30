use burn::{
    module::parallel::{
        BackwardSignal, DistributedLayer, LayerParallelism, LayerPlacement, LayerSequence,
    },
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        attention::SeqLengthOption,
        transformer::{TransformerEncoderConfig, TransformerEncoderLayer},
    },
    prelude::*,
    store::BurnpackStore,
};

/// `text-classification`'s transformer reshaped to be split across devices: the two embeddings
/// become the input layer, each encoder layer a hidden layer, and the classification head the
/// output layer.
#[derive(Module, Debug)]
pub struct LayeredClassifier {
    input: Embeddings,
    hidden: Vec<EncoderLayer>,
    output: Head,
}

impl LayeredClassifier {
    /// Each layer is built on the device `placement` gives it, so the model never exists whole on
    /// one device.
    pub fn new(
        transformer: &TransformerEncoderConfig,
        seq_length: SeqLengthOption,
        vocab_size: usize,
        n_classes: usize,
        placement: &LayerPlacement,
    ) -> Self {
        let max_seq_length = match seq_length {
            SeqLengthOption::Fixed(max) | SeqLengthOption::Max(max) => max,
            SeqLengthOption::NoMax => panic!("the position embedding needs a maximum length"),
        };
        let d_model = transformer.d_model;
        Self {
            input: Embeddings {
                token: EmbeddingConfig::new(vocab_size, d_model).init(&placement.input),
                position: EmbeddingConfig::new(max_seq_length, d_model).init(&placement.input),
            },
            hidden: placement
                .hidden
                .iter()
                .map(|device| EncoderLayer {
                    layer: TransformerEncoderLayer::new(transformer, device),
                })
                .collect(),
            output: Head {
                linear: LinearConfig::new(d_model, n_classes).init(&placement.output),
            },
        }
    }

    /// Read a checkpoint saved from `text-classification`'s model under this model's keys.
    pub fn remap_from_single_device(store: BurnpackStore) -> BurnpackStore {
        store
            .with_remap_pattern(r"^embedding_token\.", "input.token.")
            .with_remap_pattern(r"^embedding_pos\.", "input.position.")
            .with_remap_pattern(r"^transformer\.layers\.(\d+)\.", "hidden.$1.layer.")
            .with_remap_pattern(r"^output\.", "output.linear.")
    }

    /// Each layer's parameter count and device.
    pub fn print_placement(&self, placement: &LayerPlacement) {
        println!("placement:");
        println!(
            "  embeddings {:>9} params  {:?}",
            self.input.num_params(),
            placement.input
        );
        for (index, device) in placement.hidden.iter().enumerate() {
            let params = self.hidden[index].num_params();
            println!("  encoder {index:<3} {params:>8} params  {device:?}");
        }
        println!(
            "  head       {:>9} params  {:?}",
            self.output.num_params(),
            placement.output
        );
    }
}

impl LayerParallelism for LayeredClassifier {
    type InputLayer = Embeddings;
    type HiddenLayer = EncoderLayer;
    type OutputLayer = Head;

    fn layer_input(&self) -> &Embeddings {
        &self.input
    }

    fn layers_hidden(&self) -> impl Iterator<Item = &EncoderLayer> {
        self.hidden.iter()
    }

    fn layer_output(&self) -> &Head {
        &self.output
    }

    fn into_layers(self) -> LayerSequence<Self> {
        LayerSequence {
            input: self.input,
            hidden: self.hidden,
            output: self.output,
        }
    }

    fn from_layers(layers: LayerSequence<Self>) -> Self {
        Self {
            input: layers.input,
            hidden: layers.hidden,
            output: layers.output,
        }
    }
}

/// What the model takes: token ids, and which of them are padding.
#[derive(Module, Debug)]
pub struct Tokens {
    pub ids: Tensor<2, Int>,
    pub mask_pad: Tensor<2, Bool>,
}

/// What passes between the layers. The padding mask travels with the hidden states, since every
/// encoder layer reads it.
#[derive(Module, BackwardSignal, Debug)]
pub struct Encoded {
    hidden: Tensor<3>,
    mask_pad: Tensor<2, Bool>,
}

#[derive(Module, Debug)]
pub struct Embeddings {
    token: Embedding,
    position: Embedding,
}

impl DistributedLayer for Embeddings {
    type Input = Tokens;
    type Output = Encoded;

    /// The token and position embeddings averaged, as `text-classification`'s model embeds.
    fn forward(&self, tokens: Tokens) -> Encoded {
        let [batch_size, seq_length] = tokens.ids.dims();
        let positions = Tensor::arange(0..seq_length as i64, &tokens.ids.device())
            .reshape([1, seq_length])
            .repeat_dim(0, batch_size);
        let hidden = (self.position.forward(positions) + self.token.forward(tokens.ids)) / 2;
        Encoded {
            hidden,
            mask_pad: tokens.mask_pad,
        }
    }
}

#[derive(Module, Debug)]
pub struct EncoderLayer {
    layer: TransformerEncoderLayer,
}

impl DistributedLayer for EncoderLayer {
    type Input = Encoded;
    type Output = Encoded;

    fn forward(&self, encoded: Encoded) -> Encoded {
        let hidden = self
            .layer
            .forward(encoded.hidden, Some(encoded.mask_pad.clone()), None);
        Encoded {
            hidden,
            mask_pad: encoded.mask_pad,
        }
    }
}

#[derive(Module, Debug)]
pub struct Head {
    linear: Linear,
}

impl DistributedLayer for Head {
    type Input = Encoded;
    type Output = Tensor<2>;

    /// The class logits, read off the first position.
    fn forward(&self, encoded: Encoded) -> Tensor<2> {
        let [batch_size, _, d_model] = encoded.hidden.dims();
        let first = encoded.hidden.slice([0..batch_size, 0..1]);
        self.linear.forward(first.reshape([batch_size, d_model]))
    }
}
