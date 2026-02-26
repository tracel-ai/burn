//! Pretrained weights loading for DISTS.

use burn_core as burn;

use burn::tensor::backend::Backend;
use burn_std::network::downloader::download_file_as_bytes;
use burn_store::{ModuleSnapshot, PytorchStore};
use std::fs::{File, create_dir_all};
use std::io::Write;
use std::path::PathBuf;

use super::metric::Dists;

/// URL for pretrained DISTS alpha/beta weights from the official repository.
/// Reference: https://github.com/dingkeyan93/DISTS
const DISTS_WEIGHTS_URL: &str =
    "https://github.com/dingkeyan93/DISTS/raw/master/DISTS_pytorch/weights/DISTS.pt";

/// URL for ImageNet pretrained VGG16 backbone weights from PyTorch.
const VGG16_IMAGENET_URL: &str = "https://download.pytorch.org/models/vgg16-397923af.pth";

/// Get the cache directory for DISTS weights.
fn get_cache_dir() -> PathBuf {
    let cache_dir = dirs::cache_dir()
        .expect("Could not get cache directory")
        .join("burn-dataset")
        .join("dists");

    if !cache_dir.exists() {
        create_dir_all(&cache_dir).expect("Failed to create cache directory");
    }

    cache_dir
}

/// Download file if not cached.
fn download_if_needed(url: &str, cache_path: &PathBuf, message: &str) {
    if !cache_path.exists() {
        let bytes = download_file_as_bytes(url, message);
        let mut file = File::create(cache_path).expect("Failed to create cache file");
        file.write_all(&bytes).expect("Failed to write weights");
    }
}

/// Download and load pretrained weights into a DISTS module.
///
/// This loads both:
/// 1. ImageNet pretrained VGG16 backbone weights
/// 2. DISTS trained alpha/beta weights
///
/// Weights are cached in the user's cache directory to avoid re-downloading.
///
/// # Arguments
///
/// * `dists` - The DISTS module to load weights into.
///
/// # Returns
///
/// The DISTS module with loaded pretrained weights.
pub fn load_pretrained_weights<B: Backend>(mut dists: Dists<B>) -> Dists<B> {
    let cache_dir = get_cache_dir();

    // Step 1: Download and load VGG16 ImageNet backbone weights
    let vgg_cache_path = cache_dir.join("vgg16_backbone.pth");
    download_if_needed(
        VGG16_IMAGENET_URL,
        &vgg_cache_path,
        "Downloading VGG16 ImageNet weights for DISTS...",
    );

    // Step 2: Download DISTS alpha/beta weights
    let dists_cache_path = cache_dir.join("dists_weights.pt");
    download_if_needed(
        DISTS_WEIGHTS_URL,
        &dists_cache_path,
        "Downloading DISTS alpha/beta weights...",
    );

    // Load VGG16 backbone weights first
    dists = load_vgg16_backbone_weights(dists, &vgg_cache_path);

    // Then load DISTS alpha/beta weights
    dists = load_dists_weights(dists, &dists_cache_path);

    dists
}

/// Load VGG16 ImageNet pretrained backbone weights.
fn load_vgg16_backbone_weights<B: Backend>(mut dists: Dists<B>, cache_path: &PathBuf) -> Dists<B> {
    let mut store = PytorchStore::from_file(cache_path)
        .allow_partial(true)
        .skip_enum_variants(true)
        // VGG16 features.X -> extractor.convY_Z
        .with_key_remapping(r"^features\.0\.", "extractor.conv1_1.")
        .with_key_remapping(r"^features\.2\.", "extractor.conv1_2.")
        .with_key_remapping(r"^features\.5\.", "extractor.conv2_1.")
        .with_key_remapping(r"^features\.7\.", "extractor.conv2_2.")
        .with_key_remapping(r"^features\.10\.", "extractor.conv3_1.")
        .with_key_remapping(r"^features\.12\.", "extractor.conv3_2.")
        .with_key_remapping(r"^features\.14\.", "extractor.conv3_3.")
        .with_key_remapping(r"^features\.17\.", "extractor.conv4_1.")
        .with_key_remapping(r"^features\.19\.", "extractor.conv4_2.")
        .with_key_remapping(r"^features\.21\.", "extractor.conv4_3.")
        .with_key_remapping(r"^features\.24\.", "extractor.conv5_1.")
        .with_key_remapping(r"^features\.26\.", "extractor.conv5_2.")
        .with_key_remapping(r"^features\.28\.", "extractor.conv5_3.");

    let result = dists.load_from(&mut store);
    if let Err(e) = result {
        log::warn!("Some VGG16 backbone weights could not be loaded: {:?}", e);
    }

    dists
}

/// Load DISTS trained alpha/beta weights.
fn load_dists_weights<B: Backend>(mut dists: Dists<B>, cache_path: &PathBuf) -> Dists<B> {
    let mut store = PytorchStore::from_file(cache_path)
        .allow_partial(true)
        .skip_enum_variants(true)
        // Alpha and beta weights
        .with_key_remapping(r"^alpha$", "alpha")
        .with_key_remapping(r"^beta$", "beta");

    let result = dists.load_from(&mut store);
    if let Err(e) = result {
        log::warn!("Some DISTS weights could not be loaded: {:?}", e);
    }

    dists
}
