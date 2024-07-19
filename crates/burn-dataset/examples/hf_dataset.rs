use burn_dataset::HuggingfaceDatasetLoader;
use burn_dataset::SqliteDataset;
use serde::Deserialize;

#[derive(Deserialize, Debug, Clone)]
struct MnistItemRaw {
    pub _image_bytes: Vec<u8>,
    pub _label: usize,
}
fn main() {
    // There are some datasets, such as https://huggingface.co/datasets/ylecun/mnist/tree/main that contains a script,
    // In this cases you must enable trusting remote code execution if you want to use it.
    let _train_ds: SqliteDataset<MnistItemRaw> = HuggingfaceDatasetLoader::new("mnist")
        .with_trust_remote_code(true)
        .dataset("train")
        .unwrap();

    // However not all dataset requires it https://huggingface.co/datasets/Anthropic/hh-rlhf/tree/main
    let _train_ds: SqliteDataset<MnistItemRaw> = HuggingfaceDatasetLoader::new("Anthropic/hh-rlhf")
        .dataset("train")
        .unwrap();
}
