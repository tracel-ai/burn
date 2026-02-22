//! Pretrained weights loading for LPIPS.

use burn_core as burn;

use burn::tensor::backend::Backend;
use burn_std::network::downloader::download_file_as_bytes;
use burn_store::{ModuleSnapshot, PytorchStore};
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::PathBuf;

use super::metric::{Lpips, LpipsNet};

/// URLs for pretrained LPIPS linear layer weights from the official repository.
/// Reference: https://github.com/richzhang/PerceptualSimilarity
const LPIPS_VGG_URL: &str =
    "https://github.com/richzhang/PerceptualSimilarity/raw/master/lpips/weights/v0.1/vgg.pth";
const LPIPS_ALEX_URL: &str =
    "https://github.com/richzhang/PerceptualSimilarity/raw/master/lpips/weights/v0.1/alex.pth";
const LPIPS_SQUEEZE_URL: &str =
    "https://github.com/richzhang/PerceptualSimilarity/raw/master/lpips/weights/v0.1/squeeze.pth";

/// URLs for ImageNet pretrained backbone weights from PyTorch.
const VGG16_IMAGENET_URL: &str = "https://download.pytorch.org/models/vgg16-397923af.pth";
const ALEXNET_IMAGENET_URL: &str = "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth";
const SQUEEZENET_IMAGENET_URL: &str =
    "https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth";

/// Get the download URL for LPIPS linear layer weights.
pub fn get_lpips_weights_url(net: LpipsNet) -> &'static str {
    match net {
        LpipsNet::Vgg => LPIPS_VGG_URL,
        LpipsNet::Alex => LPIPS_ALEX_URL,
        LpipsNet::Squeeze => LPIPS_SQUEEZE_URL,
    }
}

/// Get the download URL for backbone ImageNet weights.
pub fn get_backbone_weights_url(net: LpipsNet) -> &'static str {
    match net {
        LpipsNet::Vgg => VGG16_IMAGENET_URL,
        LpipsNet::Alex => ALEXNET_IMAGENET_URL,
        LpipsNet::Squeeze => SQUEEZENET_IMAGENET_URL,
    }
}

/// Get the cache directory for LPIPS weights.
fn get_cache_dir() -> PathBuf {
    let cache_dir = dirs::cache_dir()
        .expect("Could not get cache directory")
        .join("burn-dataset")
        .join("lpips");

    if !cache_dir.exists() {
        create_dir_all(&cache_dir).expect("Failed to create cache directory");
    }

    cache_dir
}

/// Download file if not cached and return the cache path.
fn download_if_needed(url: &str, cache_path: &PathBuf, message: &str) {
    if !cache_path.exists() {
        let bytes = download_file_as_bytes(url, message);
        let mut file = File::create(cache_path).expect("Failed to create cache file");
        file.write_all(&bytes).expect("Failed to write weights");
    }
}

/// Download and load pretrained weights into an LPIPS module.
///
/// This loads both:
/// 1. ImageNet pretrained backbone weights (VGG16/AlexNet/SqueezeNet)
/// 2. LPIPS trained linear layer weights
///
/// Weights are cached in the user's cache directory to avoid re-downloading.
///
/// # Arguments
///
/// * `lpips` - The LPIPS module to load weights into.
/// * `net` - The network type (determines which weights to download).
///
/// # Returns
///
/// The LPIPS module with loaded pretrained weights.
pub fn load_pretrained_weights<B: Backend>(mut lpips: Lpips<B>, net: LpipsNet) -> Lpips<B> {
    let cache_dir = get_cache_dir();

    // Step 1: Load backbone ImageNet weights
    let backbone_url = get_backbone_weights_url(net);
    let backbone_cache_path = cache_dir.join(format!("{:?}_backbone.pth", net).to_lowercase());
    let backbone_message = match net {
        LpipsNet::Vgg => "Downloading VGG16 ImageNet weights...",
        LpipsNet::Alex => "Downloading AlexNet ImageNet weights...",
        LpipsNet::Squeeze => "Downloading SqueezeNet ImageNet weights...",
    };
    download_if_needed(backbone_url, &backbone_cache_path, backbone_message);

    // Step 2: Load LPIPS linear layer weights
    let lpips_url = get_lpips_weights_url(net);
    let lpips_cache_path = cache_dir.join(format!("{:?}_lpips.pth", net).to_lowercase());
    let lpips_message = match net {
        LpipsNet::Vgg => "Downloading LPIPS VGG weights...",
        LpipsNet::Alex => "Downloading LPIPS AlexNet weights...",
        LpipsNet::Squeeze => "Downloading LPIPS SqueezeNet weights...",
    };
    download_if_needed(lpips_url, &lpips_cache_path, lpips_message);

    // Load backbone weights first
    lpips = load_backbone_weights(lpips, net, &backbone_cache_path);

    // Then load LPIPS linear layer weights
    lpips = load_lpips_weights(lpips, net, &lpips_cache_path);

    lpips
}

/// Load ImageNet pretrained backbone weights.
fn load_backbone_weights<B: Backend>(
    lpips: Lpips<B>,
    _net: LpipsNet,
    cache_path: &PathBuf,
) -> Lpips<B> {
    // Load directly into the inner struct to avoid enum variant issues
    match lpips {
        Lpips::Vgg(mut inner) => {
            let mut store = PytorchStore::from_file(cache_path)
                .allow_partial(true)
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
            if let Err(e) = inner.load_from(&mut store) {
                log::warn!("Some VGG backbone weights could not be loaded: {:?}", e);
            }
            Lpips::Vgg(inner)
        }
        Lpips::Alex(mut inner) => {
            let mut store = PytorchStore::from_file(cache_path)
                .allow_partial(true)
                // AlexNet features.X -> extractor.convY
                .with_key_remapping(r"^features\.0\.", "extractor.conv1.")
                .with_key_remapping(r"^features\.3\.", "extractor.conv2.")
                .with_key_remapping(r"^features\.6\.", "extractor.conv3.")
                .with_key_remapping(r"^features\.8\.", "extractor.conv4.")
                .with_key_remapping(r"^features\.10\.", "extractor.conv5.");
            if let Err(e) = inner.load_from(&mut store) {
                log::warn!("Some AlexNet backbone weights could not be loaded: {:?}", e);
            }
            Lpips::Alex(inner)
        }
        Lpips::Squeeze(mut inner) => {
            let mut store = PytorchStore::from_file(cache_path)
                .allow_partial(true)
                // SqueezeNet features.X -> extractor.*
                .with_key_remapping(r"^features\.0\.", "extractor.conv1.")
                .with_key_remapping(r"^features\.3\.", "extractor.fire1.")
                .with_key_remapping(r"^features\.4\.", "extractor.fire2.")
                .with_key_remapping(r"^features\.6\.", "extractor.fire3.")
                .with_key_remapping(r"^features\.7\.", "extractor.fire4.")
                .with_key_remapping(r"^features\.9\.", "extractor.fire5.")
                .with_key_remapping(r"^features\.10\.", "extractor.fire6.")
                .with_key_remapping(r"^features\.11\.", "extractor.fire7.")
                .with_key_remapping(r"^features\.12\.", "extractor.fire8.");
            if let Err(e) = inner.load_from(&mut store) {
                log::warn!("Some SqueezeNet backbone weights could not be loaded: {:?}", e);
            }
            Lpips::Squeeze(inner)
        }
    }
}

/// Load LPIPS trained linear layer weights.
fn load_lpips_weights<B: Backend>(
    lpips: Lpips<B>,
    _net: LpipsNet,
    cache_path: &PathBuf,
) -> Lpips<B> {
    // Load directly into the inner struct to avoid enum variant issues
    match lpips {
        Lpips::Vgg(mut inner) => {
            let mut store = PytorchStore::from_file(cache_path)
                .allow_partial(true)
                .with_key_remapping(r"^lin0\.model\.1\.", "lin0.")
                .with_key_remapping(r"^lin1\.model\.1\.", "lin1.")
                .with_key_remapping(r"^lin2\.model\.1\.", "lin2.")
                .with_key_remapping(r"^lin3\.model\.1\.", "lin3.")
                .with_key_remapping(r"^lin4\.model\.1\.", "lin4.");
            if let Err(e) = inner.load_from(&mut store) {
                log::warn!("Some VGG LPIPS weights could not be loaded: {:?}", e);
            }
            Lpips::Vgg(inner)
        }
        Lpips::Alex(mut inner) => {
            let mut store = PytorchStore::from_file(cache_path)
                .allow_partial(true)
                .with_key_remapping(r"^lin0\.model\.1\.", "lin0.")
                .with_key_remapping(r"^lin1\.model\.1\.", "lin1.")
                .with_key_remapping(r"^lin2\.model\.1\.", "lin2.")
                .with_key_remapping(r"^lin3\.model\.1\.", "lin3.")
                .with_key_remapping(r"^lin4\.model\.1\.", "lin4.");
            if let Err(e) = inner.load_from(&mut store) {
                log::warn!("Some AlexNet LPIPS weights could not be loaded: {:?}", e);
            }
            Lpips::Alex(inner)
        }
        Lpips::Squeeze(mut inner) => {
            let mut store = PytorchStore::from_file(cache_path)
                .allow_partial(true)
                .with_key_remapping(r"^lin0\.model\.1\.", "lin0.")
                .with_key_remapping(r"^lin1\.model\.1\.", "lin1.")
                .with_key_remapping(r"^lin2\.model\.1\.", "lin2.")
                .with_key_remapping(r"^lin3\.model\.1\.", "lin3.")
                .with_key_remapping(r"^lin4\.model\.1\.", "lin4.")
                .with_key_remapping(r"^lin5\.model\.1\.", "lin5.")
                .with_key_remapping(r"^lin6\.model\.1\.", "lin6.");
            if let Err(e) = inner.load_from(&mut store) {
                log::warn!("Some SqueezeNet LPIPS weights could not be loaded: {:?}", e);
            }
            Lpips::Squeeze(inner)
        }
    }
}
