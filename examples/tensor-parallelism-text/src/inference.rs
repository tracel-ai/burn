use std::{path::Path, sync::Arc};

use burn::{
    data::dataloader::batcher::Batcher,
    nn::{attention::SeqLengthOption, transformer::TransformerEncoderConfig},
    prelude::*,
    store::{BurnpackStore, ModuleStore},
    tensor::{Bytes, Tolerance},
};
use text_classification::{
    AgNewsDataset, TextClassificationDataset,
    data::{BertCasedTokenizer, TextClassificationBatcher, Tokenizer},
    model::{TextClassificationModel, TextClassificationModelConfig},
    training::ExperimentConfig,
};

use crate::split::{ShardedLoad, megatron};

/// From AG News's test set.
const SAMPLES: [&str; 3] = [
    "Jays power up to take finale Contrary to popular belief, the power never really snapped back \
     at SkyDome on Sunday. The lights came on after an hour delay, but it took some extra time for \
     the batting orders to provide some extra wattage.",
    "Yemen Sentences 15 Militants on Terror Charges A court in Yemen has sentenced one man to death \
     and 14 others to prison terms for a series of attacks and terrorist plots in 2002, including \
     the bombing of a French oil tanker.",
    "IBM puts grids to work at U.S. Open IBM will put a collection of its On Demand-related \
     products and technologies to this test next week at the U.S. Open tennis championships, \
     implementing a grid-based infrastructure capable of running multiple workloads including two \
     not associated with the tournament.",
];

/// Run the AG News classifier split over `devices` by tensor parallelism, and check it against
/// the same weights on the first device alone.
pub fn infer(devices: Vec<Device>, artifact_dir: &str) {
    let sharded = Device::shard_options(&devices)
        .init()
        .expect("the devices should form a sharded compute device");
    let members = sharded.shard_members().unwrap();
    let tokenizer = Arc::new(BertCasedTokenizer::default());
    let checkpoint = Checkpoint::new(artifact_dir, tokenizer.vocab_size(), &members[0]);
    let architecture = &checkpoint.architecture;
    let n_heads = architecture.transformer.n_heads;
    assert!(
        n_heads.is_multiple_of(members.len()),
        "{} members cannot each take whole heads of {n_heads}",
        members.len()
    );

    let mut load = ShardedLoad::new(checkpoint.store(), megatron);
    let model = architecture.classifier(&sharded).map(&mut load);
    println!(
        "each of {} members holds {:.0}% of the weights",
        members.len(),
        100.0 * load.share_per_member()
    );
    let mut single = architecture.classifier(&members[0]);
    checkpoint
        .store()
        .apply_to(&mut single)
        .expect("the checkpoint should hold every weight");

    let samples: Vec<String> = SAMPLES.map(String::from).to_vec();
    let batcher = TextClassificationBatcher::new(tokenizer, architecture.seq_length);
    let probabilities = model.infer(batcher.batch(samples.clone(), &sharded));
    println!("output: {:?}", probabilities.sharding().unwrap());

    let probabilities = probabilities.into_data();
    let expected = single.infer(batcher.batch(samples.clone(), &members[0]));
    probabilities.assert_approx_eq::<f32>(&expected.into_data(), Tolerance::permissive());
    println!("matches the single-device model\n");

    let n_classes = AgNewsDataset::num_classes();
    let probabilities = probabilities.try_to_vec::<f32>().unwrap();
    for (text, row) in samples.iter().zip(probabilities.chunks(n_classes)) {
        let (class, probability) = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap();
        let headline: String = text.chars().take(60).collect();
        println!(
            "{:<10} {probability:.2}  {headline}...",
            AgNewsDataset::class_name(class)
        );
    }
}

/// The classifier `ag-news-train` trains, without its weights.
struct Architecture {
    transformer: TransformerEncoderConfig,
    seq_length: SeqLengthOption,
    vocab_size: usize,
}

impl Architecture {
    /// The classifier on `device`, its weights not initialized until they are used or loaded.
    fn classifier(&self, device: &Device) -> TextClassificationModel {
        TextClassificationModelConfig::new(
            self.transformer.clone(),
            AgNewsDataset::num_classes(),
            self.vocab_size,
            self.seq_length,
        )
        .init(device)
    }
}

/// Where both models read their weights.
enum WeightSource {
    File(String),
    Bytes(Bytes),
}

/// The weights both models load, and the architecture they fit.
struct Checkpoint {
    architecture: Architecture,
    weights: WeightSource,
}

impl Checkpoint {
    /// `ag-news-train`'s model when `artifact_dir` holds it, or else fresh weights of the same
    /// architecture, which classify at random but split the same way.
    fn new(artifact_dir: &str, vocab_size: usize, device: &Device) -> Self {
        let config = format!("{artifact_dir}/config.json");
        if !Path::new(&config).exists() {
            println!("no trained model in {artifact_dir}: using fresh weights\n");
            return Self::fresh(vocab_size, device);
        }
        let config = ExperimentConfig::load(config).expect("ag-news-train's config should parse");
        Self {
            architecture: Architecture {
                transformer: config.transformer,
                seq_length: config.seq_length,
                vocab_size,
            },
            weights: WeightSource::File(format!("{artifact_dir}/model")),
        }
    }

    /// `ag-news-train`'s architecture, freshly initialized on `device`.
    fn fresh(vocab_size: usize, device: &Device) -> Self {
        let architecture = Architecture {
            transformer: TransformerEncoderConfig::new(128, 512, 4, 4)
                .with_norm_first(true)
                .with_quiet_softmax(true),
            seq_length: SeqLengthOption::Fixed(256),
            vocab_size,
        };
        let mut store = BurnpackStore::from_bytes(None);
        store
            .collect_from(&architecture.classifier(device))
            .expect("fresh weights should serialize");
        let bytes = store.get_bytes().expect("fresh weights should serialize");
        Self {
            architecture,
            weights: WeightSource::Bytes(bytes),
        }
    }

    /// A store that reads the weights from the start.
    fn store(&self) -> BurnpackStore {
        match &self.weights {
            WeightSource::File(path) => BurnpackStore::from_file(path),
            WeightSource::Bytes(bytes) => BurnpackStore::from_bytes(Some(bytes.clone())),
        }
    }
}
