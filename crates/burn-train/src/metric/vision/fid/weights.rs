use burn_std::network::downloader::download_file_as_bytes;
use burn_store::{ModuleSnapshot, PytorchStore};
use std::fs::{File, create_dir_all};
use std::io::Write;
use std::path::PathBuf;

use super::metric::Fid;

const INCEPTION_WEIGHTS_URL: &str = "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt-inception-2015-12-05-6726825d.pth";

fn get_cache_dir() -> PathBuf {
    let cache_dir = dirs::cache_dir()
        .expect("Could not get cache directory")
        .join("burn-dataset")
        .join("fid");

    if !cache_dir.exists() {
        create_dir_all(&cache_dir).expect("Failed to create cache directory");
    }

    cache_dir
}

fn download_if_needed(url: &str, cache_path: &PathBuf, message: &str) {
    if !cache_path.exists() {
        let bytes = download_file_as_bytes(url, message);
        let mut file = File::create(cache_path).expect("Failed to create cache file");
        file.write_all(&bytes).expect("Failed to write weights");
    }
}

/// Download and load pretrained pytorch-fid InceptionV3 weights.
///
/// Weights are cached in `~/.cache/burn-dataset/fid/`.
pub fn load_pretrained_weights(mut fid: Fid) -> Fid {
    let cache_dir = get_cache_dir();
    let cache_path = cache_dir.join("pt-inception-2015-12-05-6726825d.pth");

    download_if_needed(
        INCEPTION_WEIGHTS_URL,
        &cache_path,
        "Downloading InceptionV3 weights for FID...",
    );

    // PyTorch keys use "Conv2d_1a_3x3.*" etc, our model nests them under "extractor.*"
    let mut store = PytorchStore::from_file(&cache_path)
        .allow_partial(true)
        .with_key_remapping(r"^Conv2d_1a_3x3\.", "extractor.conv2d_1a.")
        .with_key_remapping(r"^Conv2d_2a_3x3\.", "extractor.conv2d_2a.")
        .with_key_remapping(r"^Conv2d_2b_3x3\.", "extractor.conv2d_2b.")
        .with_key_remapping(r"^Conv2d_3b_1x1\.", "extractor.conv2d_3b.")
        .with_key_remapping(r"^Conv2d_4a_3x3\.", "extractor.conv2d_4a.")
        .with_key_remapping(r"^Mixed_5b\.", "extractor.mixed_5b.")
        .with_key_remapping(r"^Mixed_5c\.", "extractor.mixed_5c.")
        .with_key_remapping(r"^Mixed_5d\.", "extractor.mixed_5d.")
        .with_key_remapping(r"^Mixed_6a\.", "extractor.mixed_6a.")
        .with_key_remapping(r"^Mixed_6b\.", "extractor.mixed_6b.")
        .with_key_remapping(r"^Mixed_6c\.", "extractor.mixed_6c.")
        .with_key_remapping(r"^Mixed_6d\.", "extractor.mixed_6d.")
        .with_key_remapping(r"^Mixed_6e\.", "extractor.mixed_6e.")
        .with_key_remapping(r"^Mixed_7a\.", "extractor.mixed_7a.")
        .with_key_remapping(r"^Mixed_7b\.", "extractor.mixed_7b.")
        .with_key_remapping(r"^Mixed_7c\.", "extractor.mixed_7c.");

    if let Err(e) = fid.load_from(&mut store) {
        log::warn!("Some InceptionV3 weights could not be loaded: {:?}", e);
    }

    fid
}
