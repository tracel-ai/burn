use burn_core as burn;

use super::vgg19::Vgg19;
use burn::tensor::Device;
use burn_core::network::downloader::download_file_as_bytes;
use burn_store::{ModuleSnapshot, PytorchStore};
use std::fs::{File, create_dir_all, rename};
use std::io::Write;
use std::path::PathBuf;

const VGG19_URL: &str = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth";

/// Resolves and returns the local cache directory for the VGG19 weights.
///
/// Creates the directory `~/.cache/burn-pretrained-models/loss/vgg19/`
/// (or OS equivalent) if it does not already exist.
fn get_cache_dir() -> PathBuf {
    let cache_dir = dirs::cache_dir()
        .expect("Failed to get cache directory for Gram Matrix Loss")
        .join("burn-pretrained-models")
        .join("loss")
        .join("vgg19");

    if !cache_dir.exists() {
        create_dir_all(&cache_dir).expect("Failed to create cache directory for Gram Matrix Loss");
    }

    cache_dir
}

/// Downloads the pretrained weights to the `cache_path` if they don't exist already.
///
/// Requires an active internet connection on the first run. Subsequent runs will
/// use the locally cached `.pth` file.
fn download_weights_if_not_saved(cache_path: &PathBuf) {
    if !cache_path.exists() {
        let bytes = download_file_as_bytes(
            VGG19_URL,
            "Downloading VGG19 ImageNet weights for Gram Matrix Loss...",
        );

        // Write to a temporary file. If writing gets completed, then rename to the actual/correct name.
        // If writing is not completed, the file with the correct name (i.e. `cache_path`) will not exist
        // so this code block can run again which is the desired behavior.
        let temp_path = cache_path.with_extension("pth.tmp");
        let mut file = File::create(&temp_path)
            .expect("Failed to create VGG19 cache file for Gram Matrix Loss");
        file.write_all(&bytes)
            .expect("Failed to write VGG19 weights to the cache file for Gram Matrix Loss");

        rename(temp_path, cache_path)
            .expect("Failed to rename temporary file to the actual VGG19 cache file name for Gram Matrix Loss");
    }
}

/// Loads ImageNet pretrained weights into the provided VGG19 feature extractor.
///
/// This function downloads the official PyTorch VGG19 weights, remaps the keys
/// from PyTorch's `features.X` format to Burn's `convX_Y` format, and loads
/// them into the module.
///
/// # Arguments
///
/// - `vgg19` - An initialized VGG19 module with random weights.
///
/// # Returns
///
/// The VGG19 module with pretrained ImageNet weights loaded.
pub fn load_vgg19_weights(mut vgg19: Vgg19) -> Vgg19 {
    let cache_dir = get_cache_dir();
    let cache_path = cache_dir.join("vgg19.pth");
    download_weights_if_not_saved(&cache_path);

    // Download the pretrained weights from PyTorch
    let mut store = PytorchStore::from_file(cache_path)
        .allow_partial(true)
        // Block 1
        .with_key_remapping(r"^features\.0\.", "conv1_1.")
        .with_key_remapping(r"^features\.2\.", "conv1_2.")
        // Block 2
        .with_key_remapping(r"^features\.5\.", "conv2_1.")
        .with_key_remapping(r"^features\.7\.", "conv2_2.")
        // Block 3
        .with_key_remapping(r"^features\.10\.", "conv3_1.")
        .with_key_remapping(r"^features\.12\.", "conv3_2.")
        .with_key_remapping(r"^features\.14\.", "conv3_3.")
        .with_key_remapping(r"^features\.16\.", "conv3_4.")
        // Block 4
        .with_key_remapping(r"^features\.19\.", "conv4_1.")
        .with_key_remapping(r"^features\.21\.", "conv4_2.")
        .with_key_remapping(r"^features\.23\.", "conv4_3.")
        .with_key_remapping(r"^features\.25\.", "conv4_4.")
        // Block 5
        .with_key_remapping(r"^features\.28\.", "conv5_1.");

    let result = vgg19.load_from(&mut store);
    if let Err(e) = result {
        eprintln!("Warning: Some VGG19 weights could not be loaded: {:?}", e);
    }

    vgg19
}
