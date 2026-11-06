use burn::{
    module::{ModuleMapper, Param},
    prelude::*,
    store::{BurnpackStore, ModuleStore},
    tensor::{
        Sharding::{self, Replicated, Split},
        Shards,
        shard::DimSplit,
    },
};

/// Projections split by output column: each member computes whole attention heads, or a slice of
/// the feed-forward's hidden width, with nothing sent between members.
const COLUMN_SPLIT: [&str; 4] = ["mha.query", "mha.key", "mha.value", "pwff.linear_inner"];

/// Projections split by input row, each taking the slice its column-split partner produced. Their
/// output is a partial sum, reduced by one all-reduce when the next op reads it.
const ROW_SPLIT: [&str; 2] = ["mha.output", "pwff.linear_outer"];

/// Megatron's split of a transformer encoder, by parameter path. A [`Linear`](burn::nn::Linear)
/// weight is `[d_input, d_output]`. A row-split projection's bias is added after the reduction, so
/// it stays replicated, as does everything else: the embeddings, the layer norms and the head.
pub fn megatron(path: &str) -> Sharding {
    let Some((module, param)) = path.rsplit_once('.') else {
        return Replicated;
    };
    let column = COLUMN_SPLIT.iter().any(|name| module.ends_with(name));
    let row = ROW_SPLIT.iter().any(|name| module.ends_with(name));
    match param {
        "weight" if column => Split { dim: 1 },
        "bias" if column => Split { dim: 0 },
        "weight" if row => Split { dim: 0 },
        _ => Replicated,
    }
}

/// Loads a checkpoint into a module built on a sharded compute device, splitting each weight as
/// `plan` says for its path. A split weight reaches each member as that member's chunk alone.
pub struct ShardedLoad<P> {
    store: BurnpackStore,
    plan: P,
    path: Vec<String>,
    whole: usize,
    per_member: usize,
}

impl<P: Fn(&str) -> Sharding> ShardedLoad<P> {
    pub fn new(store: BurnpackStore, plan: P) -> Self {
        Self {
            store,
            plan,
            path: Vec::new(),
            whole: 0,
            per_member: 0,
        }
    }

    /// The share of the model's weights each member holds.
    pub fn share_per_member(&self) -> f64 {
        self.per_member as f64 / self.whole as f64
    }

    /// The checkpoint tensor at `path` on `device`, cut into one chunk per member when split.
    fn tensor<const D: usize>(&mut self, path: &str, device: &Device) -> Tensor<D> {
        let data = self
            .store
            .get_tensor(path)
            .expect("the checkpoint should read")
            .unwrap_or_else(|| panic!("the checkpoint has no {path}"))
            .to_data()
            .expect("the checkpoint should read");
        let members = device
            .shard_members()
            .expect("the module should be on a sharded compute device");
        self.whole += data.num_elements();

        match (self.plan)(path) {
            Split { dim } => {
                self.per_member += data.num_elements() / members.len();
                let split = DimSplit {
                    dim,
                    len: data.shape()[dim],
                    members: members.len(),
                };
                let shards = split
                    .data_chunks(&data)
                    .zip(&members)
                    .map(|(chunk, member)| Tensor::from_data(chunk, member))
                    .collect();
                Tensor::from_shards(
                    Shards {
                        shards,
                        sharding: Split { dim },
                    },
                    device,
                )
                .expect("each chunk is its member's part")
            }
            _ => {
                self.per_member += data.num_elements();
                Tensor::from_data(data, device)
            }
        }
    }
}

impl<P: Fn(&str) -> Sharding> ModuleMapper for ShardedLoad<P> {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let path = self.path.join(".");
        let tensor = self.tensor(&path, &param.lazy_device());
        let id = param.id;
        param.transform_for_load(tensor, id)
    }
}
