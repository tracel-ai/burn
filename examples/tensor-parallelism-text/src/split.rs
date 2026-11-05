use burn::{
    module::Param,
    nn::{
        Linear,
        attention::MultiHeadAttention,
        transformer::{PositionWiseFeedForward, TransformerEncoderLayer},
    },
    prelude::*,
    store::{BurnpackStore, ModuleSnapshot, ModuleStore, PathFilter},
    tensor::{Sharding::Split, Shards, shard::DimSplit},
};
use text_classification::model::TextClassificationModel;

/// A checkpoint read into a model on a sharded compute device: the weights the model's
/// [`TensorParallel`] plan splits reach each member as that member's chunk alone, and every other
/// weight loads whole on every member.
pub struct ShardedLoad {
    store: BurnpackStore,
    device: Device,
    members: Vec<Device>,
    split_paths: Vec<String>,
}

impl ShardedLoad {
    /// # Panics
    ///
    /// When `device` is not a sharded compute device.
    pub fn new(store: BurnpackStore, device: &Device) -> Self {
        let members = device
            .shard_members()
            .expect("tensor parallelism needs a sharded compute device");
        Self {
            store,
            device: device.clone(),
            members,
            split_paths: Vec::new(),
        }
    }

    /// `model` with every weight loaded, split as its plan says.
    pub fn load<M: TensorParallel + ModuleSnapshot>(mut self, model: M) -> M {
        let mut model = model.tensor_parallel(&mut self, "");
        let replicated: Vec<String> = self
            .store
            .keys()
            .expect("the checkpoint should list its tensors")
            .into_iter()
            .filter(|path| !self.split_paths.contains(path))
            .collect();
        self.store
            .filter(PathFilter::new().with_full_paths(replicated))
            .apply_to(&mut model)
            .expect("the checkpoint should hold every replicated weight");
        model
    }

    /// The weight at `path` split along `dim`. Each member's chunk is cut on the host and made on
    /// that member, so no member ever holds another's.
    fn split<const D: usize>(
        &mut self,
        param: Param<Tensor<D>>,
        path: String,
        dim: usize,
    ) -> Param<Tensor<D>> {
        let data = self
            .store
            .get_tensor(&path)
            .expect("the checkpoint should read")
            .unwrap_or_else(|| panic!("the checkpoint has no {path}"))
            .to_data()
            .expect("the checkpoint should read");
        let split = DimSplit {
            dim,
            len: data.shape()[dim],
            members: self.members.len(),
        };
        let shards = split
            .data_chunks(&data)
            .zip(&self.members)
            .map(|(chunk, member)| Tensor::from_data(chunk, member))
            .collect();
        let tensor = Tensor::from_shards(
            Shards {
                shards,
                sharding: Split { dim },
            },
            &self.device,
        )
        .expect("each chunk is its member's part");
        self.split_paths.push(path);
        Param::initialized(param.id, tensor)
    }
}

/// Megatron's split of a module over a sharded compute device: which of its weights split, and
/// along which dim. Anything a plan leaves alone, such as the embeddings, the layer norms and the
/// classification head, stays replicated: every member holds all of it.
pub trait TensorParallel {
    /// The module with the weights its plan splits read from `load`, under the checkpoint path
    /// `prefix`, which is empty or ends with a dot.
    fn tensor_parallel(self, load: &mut ShardedLoad, prefix: &str) -> Self;
}

impl TensorParallel for TextClassificationModel {
    fn tensor_parallel(mut self, load: &mut ShardedLoad, prefix: &str) -> Self {
        self.transformer.layers = self
            .transformer
            .layers
            .into_iter()
            .enumerate()
            .map(|(index, layer)| {
                layer.tensor_parallel(load, &format!("{prefix}transformer.layers.{index}."))
            })
            .collect();
        self
    }
}

impl TensorParallel for TransformerEncoderLayer {
    fn tensor_parallel(mut self, load: &mut ShardedLoad, prefix: &str) -> Self {
        self.mha = self.mha.tensor_parallel(load, &format!("{prefix}mha."));
        self.pwff = self.pwff.tensor_parallel(load, &format!("{prefix}pwff."));
        self
    }
}

/// Each member computes whole heads: the query, key and value projections are split by output
/// column, which is by head, and the output projection by input row.
impl TensorParallel for MultiHeadAttention {
    fn tensor_parallel(mut self, load: &mut ShardedLoad, prefix: &str) -> Self {
        self.query = self.query.split_columns(load, &format!("{prefix}query."));
        self.key = self.key.split_columns(load, &format!("{prefix}key."));
        self.value = self.value.split_columns(load, &format!("{prefix}value."));
        self.output = self.output.split_rows(load, &format!("{prefix}output."));
        self
    }
}

/// Each member computes a slice of the hidden width, so the activation between the two
/// projections never leaves its member.
impl TensorParallel for PositionWiseFeedForward {
    fn tensor_parallel(mut self, load: &mut ShardedLoad, prefix: &str) -> Self {
        self.linear_inner = self
            .linear_inner
            .split_columns(load, &format!("{prefix}linear_inner."));
        self.linear_outer = self
            .linear_outer
            .split_rows(load, &format!("{prefix}linear_outer."));
        self
    }
}

/// How a [`Linear`] splits. Its weight is `[d_input, d_output]`.
pub trait LinearSplit {
    /// Each member holds a slice of the output features, and so produces that slice of the
    /// output, with nothing sent between members.
    fn split_columns(self, load: &mut ShardedLoad, prefix: &str) -> Self;

    /// Each member holds a slice of the input features and takes that slice of the input. Its
    /// output is a partial sum, reduced by one all-reduce when the next op needs it whole. The
    /// bias stays replicated, added once after that reduction.
    fn split_rows(self, load: &mut ShardedLoad, prefix: &str) -> Self;
}

impl LinearSplit for Linear {
    fn split_columns(mut self, load: &mut ShardedLoad, prefix: &str) -> Self {
        self.weight = load.split(self.weight, format!("{prefix}weight"), 1);
        self.bias = self
            .bias
            .map(|bias| load.split(bias, format!("{prefix}bias"), 0));
        self
    }

    fn split_rows(mut self, load: &mut ShardedLoad, prefix: &str) -> Self {
        self.weight = load.split(self.weight, format!("{prefix}weight"), 0);
        self
    }
}
