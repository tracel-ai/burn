//! Pretrained weights loading for A-FINE.
//!
//! `afine.pth` is published by the PyIQA project on Hugging Face and
//! bundles all six A-FINE shards into a single file: a fine-tuned CLIP
//! ViT-B/32 (`finetuned_clip`), the naturalness head (`natural`), the
//! fidelity head (`fidelity`), the two per-head logistic calibrators
//! (`natural_scale`, `fidelity_scale`), and the adapter (`adapter`).
//! Each shard is loaded into the corresponding submodule of [`Afine`]
//! via a separate [`PytorchStore`] with `with_top_level_key`.
//!
//! The OpenAI CLIP `ViT-B-32.pt` checkpoint is **not** used. That file
//! is a TorchScript archive that `burn-store` cannot read, and PyIQA's
//! `finetuned_clip` shard is the variant A-FINE was actually trained
//! with — substituting stock CLIP would silently produce wrong scores.
//!
//! ## Hosting
//!
//! The URL points at the PyIQA author's personal HF account, which is
//! the canonical source for these weights. There is no organization-
//! hosted mirror anywhere (HF orgs, Zenodo, figshare, ModelScope, OSF,
//! Kaggle, GitHub releases — all checked). The hosting question is
//! flagged in the PR description for the maintainer; if a different
//! mirror is preferred, swap the URL constant below.

use burn_core as burn;

use burn::module::Param;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_std::network::downloader::download_file_as_bytes;
use burn_store::pytorch::PytorchReader;
use burn_store::{ModuleSnapshot, PytorchStore};
use std::fs::{File, create_dir_all};
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

use super::calibrators::{AfineAdapter, FrCalibratorWithLimit, NrCalibrator};
use super::metric::Afine;

/// Source URL for `afine.pth` on Hugging Face.
const AFINE_URL: &str =
    "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/afine.pth";

/// Cached filename inside `~/.cache/burn-dataset/afine/`.
const CACHE_FILENAME: &str = "afine.pth";

fn get_cache_dir() -> PathBuf {
    let cache_dir = dirs::cache_dir()
        .expect("Could not get cache directory")
        .join("burn-dataset")
        .join("afine");
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

/// Download `afine.pth` (if not cached) and load all six shards into
/// the matching submodules of an `Afine<B>` previously produced by
/// `AfineConfig::init`.
///
/// Errors during loading are logged via `log::warn!` and the function
/// returns the module unconditionally — `allow_partial(true)` plus
/// per-shard regex remapping mean unmapped or unknown checkpoint keys
/// are silently dropped, matching the behaviour of LPIPS, DISTS, and
/// FID's pretrained loaders.
pub(crate) fn load_pretrained_weights<B: Backend>(mut afine: Afine<B>) -> Afine<B> {
    let cache_dir = get_cache_dir();
    let cache_path = cache_dir.join(CACHE_FILENAME);
    download_if_needed(AFINE_URL, &cache_path, "Downloading A-FINE weights...");

    afine.clip_visual = load_clip_shard(afine.clip_visual, &cache_path);
    afine.qhead = load_qhead_shard(afine.qhead, &cache_path);
    afine.dhead = load_simple_shard(afine.dhead, &cache_path, "fidelity", "fidelity head");

    // The calibrator and adapter shards store yita3, yita4, and k as
    // 0-D scalar tensors (`shape=()`). Burn's `Param<Tensor<B, 1>>` of
    // length 1 expects shape `(1,)`, and PytorchStore drops keys whose
    // shape doesn't match the destination — silently leaving the
    // params at random init. Load these manually via PytorchReader so
    // the 0-D scalars become 1-element 1-D tensors.
    let device = afine.adapter.k.val().device();
    afine.nr_calibrator = load_nr_calibrator_scalars(&cache_path, &device);
    afine.fr_calibrator = load_fr_calibrator_scalars(&cache_path, &device);
    afine.adapter = load_adapter_scalar(&cache_path, &device);

    afine
}

/// Read a 0-D scalar value from a checkpoint shard, returning it as an
/// `f32`. Logs a warning and returns `None` on failure so the caller can
/// fall back to a default.
fn read_scalar_value<P: AsRef<Path>>(
    cache_path: P,
    top_level_key: &str,
    field_name: &str,
) -> Option<f32> {
    let reader = PytorchReader::with_top_level_key(cache_path.as_ref(), top_level_key)
        .map_err(|e| log::warn!("Failed to open shard '{top_level_key}': {e:?}"))
        .ok()?;
    let snapshot = reader.get(field_name)?;
    let data = snapshot
        .to_data()
        .map_err(|e| log::warn!("Failed to read '{top_level_key}.{field_name}' tensor data: {e:?}"))
        .ok()?;
    let values = data
        .to_vec::<f32>()
        .map_err(|e| log::warn!("Failed to convert '{top_level_key}.{field_name}' to f32: {e:?}"))
        .ok()?;
    values.first().copied()
}

fn scalar_param<B: Backend>(value: f32, device: &B::Device) -> Param<Tensor<B, 1>> {
    Param::from_tensor(Tensor::from_floats([value], device))
}

fn load_nr_calibrator_scalars<B: Backend>(
    cache_path: &PathBuf,
    device: &B::Device,
) -> NrCalibrator<B> {
    // PyIQA defaults if reading fails — preserves random-init fallback.
    const NR_YITA3_FALLBACK: f32 = 4.9592;
    const NR_YITA4_FALLBACK: f32 = 21.5968;
    let yita3 =
        read_scalar_value(cache_path, "natural_scale", "yita3").unwrap_or(NR_YITA3_FALLBACK);
    let yita4 =
        read_scalar_value(cache_path, "natural_scale", "yita4").unwrap_or(NR_YITA4_FALLBACK);
    NrCalibrator {
        yita3: scalar_param(yita3, device),
        yita4: scalar_param(yita4, device),
    }
}

fn load_fr_calibrator_scalars<B: Backend>(
    cache_path: &PathBuf,
    device: &B::Device,
) -> FrCalibratorWithLimit<B> {
    const FR_YITA3_FALLBACK: f32 = 0.5;
    const FR_YITA4_FALLBACK: f32 = 0.15;
    let yita3 =
        read_scalar_value(cache_path, "fidelity_scale", "yita3").unwrap_or(FR_YITA3_FALLBACK);
    let yita4 =
        read_scalar_value(cache_path, "fidelity_scale", "yita4").unwrap_or(FR_YITA4_FALLBACK);
    FrCalibratorWithLimit {
        yita3: scalar_param(yita3, device),
        yita4: scalar_param(yita4, device),
    }
}

fn load_adapter_scalar<B: Backend>(cache_path: &PathBuf, device: &B::Device) -> AfineAdapter<B> {
    const K_FALLBACK: f32 = 5.0;
    let k = read_scalar_value(cache_path, "adapter", "k").unwrap_or(K_FALLBACK);
    AfineAdapter {
        k: scalar_param(k, device),
    }
}

/// Load the `finetuned_clip` shard into the visual encoder.
///
/// Key-remap rules strip the `visual.` prefix used in the CLIP state
/// dict and rename the few fields where the burn module diverges from
/// PyTorch's CLIP layout: `conv1` → `patch_embed`,
/// `class_embedding` → `class_token`, `transformer.resblocks` → `blocks`,
/// and the fused `attn.in_proj_*` → `attn.qkv_proj.*`. The
/// `weight`/`bias` ↔ `gamma`/`beta` rename for LayerNorm is handled
/// automatically by `PyTorchToBurnAdapter`.
///
/// `with_regex(r"^visual\.")` filters the load to only the visual
/// encoder; the CLIP text encoder (~150 keys) is dropped at the filter
/// stage. `visual.proj` (the unused joint-embedding projection) has
/// no corresponding burn field and is silently dropped via
/// `allow_partial(true)`.
fn load_clip_shard<B: Backend>(
    mut clip: super::clip_vit::ClipVisualEncoder<B>,
    cache_path: &PathBuf,
) -> super::clip_vit::ClipVisualEncoder<B> {
    let mut store = PytorchStore::from_file(cache_path)
        .with_top_level_key("finetuned_clip")
        .allow_partial(true)
        .skip_enum_variants(true)
        // No `with_regex` filter: PathFilter matches against burn-side
        // destination paths, not the source PyTorch keys, so a regex on
        // `^visual\.` rejects every remapped key. Text-encoder keys
        // (token_embedding, transformer.resblocks.*) have no
        // corresponding burn fields and are dropped by
        // `allow_partial(true)`, which is the same end result.
        //
        // Special case: the text encoder also has a top-level
        // `positional_embedding` key (shape `[77, 512]`) which collides
        // with the visual one after our `^visual\.positional_embedding$`
        // rename targets `positional_embedding`. Without this first
        // rule, HashMap iteration order decides which one wins, giving
        // a flaky load. Rename the text encoder's first to a key that
        // matches no burn field.
        .with_key_remapping(r"^positional_embedding$", "_text_positional_embedding_drop")
        .with_key_remapping(r"^visual\.conv1\.", "patch_embed.")
        .with_key_remapping(r"^visual\.class_embedding$", "class_token")
        .with_key_remapping(r"^visual\.positional_embedding$", "positional_embedding")
        .with_key_remapping(r"^visual\.ln_pre\.", "ln_pre.")
        .with_key_remapping(r"^visual\.ln_post\.", "ln_post.")
        .with_key_remapping(r"^visual\.transformer\.resblocks\.", "blocks.")
        .with_key_remapping(r"\.attn\.in_proj_weight$", ".attn.qkv_proj.weight")
        .with_key_remapping(r"\.attn\.in_proj_bias$", ".attn.qkv_proj.bias");
    if let Err(e) = clip.load_from(&mut store) {
        log::warn!(
            "Some CLIP visual encoder weights could not be loaded: {:?}",
            e
        );
    }
    clip
}

/// Load the `natural` shard into the naturalness head.
///
/// PyIQA stores the proj_head as a `nn.Sequential` with index-2 GELU,
/// so the FC layers come out as `proj_head.0.*` and `proj_head.2.*`.
/// We named them explicitly to avoid the GELU-index gap, hence the
/// remap.
fn load_qhead_shard<B: Backend>(
    mut qhead: super::heads::AfineQHead<B>,
    cache_path: &PathBuf,
) -> super::heads::AfineQHead<B> {
    let mut store = PytorchStore::from_file(cache_path)
        .with_top_level_key("natural")
        .allow_partial(true)
        .skip_enum_variants(true)
        .with_key_remapping(r"^proj_head\.0\.", "proj_head_fc1.")
        .with_key_remapping(r"^proj_head\.2\.", "proj_head_fc2.");
    if let Err(e) = qhead.load_from(&mut store) {
        log::warn!("Some naturalness head weights could not be loaded: {:?}", e);
    }
    qhead
}

/// Load any shard whose keys map directly to burn-side field names.
/// Used for the fidelity head, both calibrators, and the adapter.
fn load_simple_shard<B, M>(
    mut module: M,
    cache_path: &PathBuf,
    top_level_key: &'static str,
    description: &str,
) -> M
where
    B: Backend,
    M: ModuleSnapshot<B>,
{
    let mut store = PytorchStore::from_file(cache_path)
        .with_top_level_key(top_level_key)
        .allow_partial(true)
        .skip_enum_variants(true);
    if let Err(e) = module.load_from(&mut store) {
        log::warn!("Some {} weights could not be loaded: {:?}", description, e);
    }
    module
}
