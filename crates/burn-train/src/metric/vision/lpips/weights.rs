//! Pretrained weights loading for LPIPS.

use burn_core as burn;

use burn::tensor::backend::Backend;
use burn_std::network::downloader::download_file_as_bytes;
use burn_store::{ModuleSnapshot, PytorchStore};
use std::io::Write;

use super::metric::{Lpips, LpipsNet};

/// URLs for pretrained LPIPS weights from the official repository.
/// Reference: https://github.com/richzhang/PerceptualSimilarity
const LPIPS_VGG_URL: &str =
    "https://github.com/richzhang/PerceptualSimilarity/raw/master/lpips/weights/v0.1/vgg.pth";
const LPIPS_ALEX_URL: &str =
    "https://github.com/richzhang/PerceptualSimilarity/raw/master/lpips/weights/v0.1/alex.pth";
const LPIPS_SQUEEZE_URL: &str =
    "https://github.com/richzhang/PerceptualSimilarity/raw/master/lpips/weights/v0.1/squeeze.pth";

/// Get the download URL for a given LPIPS network type.
pub fn get_weights_url(net: LpipsNet) -> &'static str {
    match net {
        LpipsNet::Vgg => LPIPS_VGG_URL,
        LpipsNet::Alex => LPIPS_ALEX_URL,
        LpipsNet::Squeeze => LPIPS_SQUEEZE_URL,
    }
}

/// Download and load pretrained weights into an LPIPS module.
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
    let url = get_weights_url(net);
    let message = match net {
        LpipsNet::Vgg => "Downloading LPIPS VGG weights...",
        LpipsNet::Alex => "Downloading LPIPS AlexNet weights...",
        LpipsNet::Squeeze => "Downloading LPIPS SqueezeNet weights...",
    };

    // Download weights (synchronous, tokio runtime handled internally)
    let bytes = download_file_as_bytes(url, message);

    // Write to temporary file (PytorchStore requires file path)
    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join(format!("lpips_{:?}.pth", net));
    let mut file = std::fs::File::create(&temp_path).expect("Failed to create temp file");
    file.write_all(&bytes).expect("Failed to write weights");
    drop(file);

    // Create PyTorch store with key remapping for Burn compatibility
    let mut store = PytorchStore::from_file(&temp_path)
        .allow_partial(true)
        .skip_enum_variants(true)
        // VGG feature extractor remapping
        .with_key_remapping(r"^net\.slice1\.0\.", "extractor.conv1_1.")
        .with_key_remapping(r"^net\.slice1\.2\.", "extractor.conv1_2.")
        .with_key_remapping(r"^net\.slice2\.0\.", "extractor.conv2_1.")
        .with_key_remapping(r"^net\.slice2\.2\.", "extractor.conv2_2.")
        .with_key_remapping(r"^net\.slice3\.0\.", "extractor.conv3_1.")
        .with_key_remapping(r"^net\.slice3\.2\.", "extractor.conv3_2.")
        .with_key_remapping(r"^net\.slice3\.4\.", "extractor.conv3_3.")
        .with_key_remapping(r"^net\.slice4\.0\.", "extractor.conv4_1.")
        .with_key_remapping(r"^net\.slice4\.2\.", "extractor.conv4_2.")
        .with_key_remapping(r"^net\.slice4\.4\.", "extractor.conv4_3.")
        .with_key_remapping(r"^net\.slice5\.0\.", "extractor.conv5_1.")
        .with_key_remapping(r"^net\.slice5\.2\.", "extractor.conv5_2.")
        .with_key_remapping(r"^net\.slice5\.4\.", "extractor.conv5_3.")
        // AlexNet feature extractor remapping
        .with_key_remapping(r"^net\.slice1\.0\.", "extractor.conv1.")
        .with_key_remapping(r"^net\.slice2\.0\.", "extractor.conv2.")
        .with_key_remapping(r"^net\.slice3\.0\.", "extractor.conv3.")
        .with_key_remapping(r"^net\.slice4\.0\.", "extractor.conv4.")
        .with_key_remapping(r"^net\.slice5\.0\.", "extractor.conv5.")
        // SqueezeNet feature extractor remapping
        .with_key_remapping(r"^net\.slice1\.0\.", "extractor.conv1.")
        .with_key_remapping(r"^net\.slice2\.1\.", "extractor.fire1.")
        .with_key_remapping(r"^net\.slice2\.2\.", "extractor.fire2.")
        .with_key_remapping(r"^net\.slice3\.0\.", "extractor.fire3.")
        .with_key_remapping(r"^net\.slice3\.1\.", "extractor.fire4.")
        .with_key_remapping(r"^net\.slice4\.0\.", "extractor.fire5.")
        .with_key_remapping(r"^net\.slice4\.1\.", "extractor.fire6.")
        .with_key_remapping(r"^net\.slice5\.0\.", "extractor.fire7.")
        .with_key_remapping(r"^net\.slice5\.1\.", "extractor.fire8.")
        // Linear layers remapping (common for all networks)
        .with_key_remapping(r"^lin0\.model\.1\.", "lin0.")
        .with_key_remapping(r"^lin1\.model\.1\.", "lin1.")
        .with_key_remapping(r"^lin2\.model\.1\.", "lin2.")
        .with_key_remapping(r"^lin3\.model\.1\.", "lin3.")
        .with_key_remapping(r"^lin4\.model\.1\.", "lin4.")
        .with_key_remapping(r"^lin5\.model\.1\.", "lin5.")
        .with_key_remapping(r"^lin6\.model\.1\.", "lin6.");

    // Load weights into the model
    let result = lpips.load_from(&mut store);
    if let Err(e) = result {
        log::warn!("Some weights could not be loaded: {:?}", e);
    }

    // Cleanup temp file
    let _ = std::fs::remove_file(&temp_path);

    lpips
}
