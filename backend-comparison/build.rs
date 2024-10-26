use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

const MODELS_REPO: &str = "https://github.com/tracel-ai/models.git";

// Patch resnet code (remove pretrained feature code)
const PATCH: &str = r#"diff --git a/resnet-burn/resnet/src/resnet.rs b/resnet-burn/resnet/src/resnet.rs
index e7f8787..3967049 100644
--- a/resnet-burn/resnet/src/resnet.rs
+++ b/resnet-burn/resnet/src/resnet.rs
@@ -12,13 +12,6 @@ use burn::{
 
 use super::block::{LayerBlock, LayerBlockConfig};
 
-#[cfg(feature = "pretrained")]
-use {
-    super::weights::{self, WeightsMeta},
-    burn::record::{FullPrecisionSettings, Recorder, RecorderError},
-    burn_import::pytorch::{LoadArgs, PyTorchFileRecorder},
-};
-
 // ResNet residual layer block configs
 const RESNET18_BLOCKS: [usize; 4] = [2, 2, 2, 2];
 const RESNET34_BLOCKS: [usize; 4] = [3, 4, 6, 3];
@@ -77,29 +70,6 @@ impl<B: Backend> ResNet<B> {
         ResNetConfig::new(RESNET18_BLOCKS, num_classes, 1).init(device)
     }

-    /// ResNet-18 from [`Deep Residual Learning for Image Recognition`](https://arxiv.org/abs/1512.03385)
-    /// with pre-trained weights.
-    ///
-    /// # Arguments
-    ///
-    /// * `weights`: Pre-trained weights to load.
-    /// * `device` - Device to create the module on.
-    ///
-    /// # Returns
-    ///
-    /// A ResNet-18 module with pre-trained weights.
-    #[cfg(feature = "pretrained")]
-    pub fn resnet18_pretrained(
-        weights: weights::ResNet18,
-        device: &Device<B>,
-    ) -> Result<Self, RecorderError> {
-        let weights = weights.weights();
-        let record = Self::load_weights_record(&weights, device)?;
-        let model = ResNet::<B>::resnet18(weights.num_classes, device).load_record(record);
-
-        Ok(model)
-    }
-
     /// ResNet-34 from [`Deep Residual Learning for Image Recognition`](https://arxiv.org/abs/1512.03385).
     ///
     /// # Arguments
@@ -114,29 +84,6 @@ impl<B: Backend> ResNet<B> {
         ResNetConfig::new(RESNET34_BLOCKS, num_classes, 1).init(device)
     }

-    /// ResNet-34 from [`Deep Residual Learning for Image Recognition`](https://arxiv.org/abs/1512.03385)
-    /// with pre-trained weights.
-    ///
-    /// # Arguments
-    ///
-    /// * `weights`: Pre-trained weights to load.
-    /// * `device` - Device to create the module on.
-    ///
-    /// # Returns
-    ///
-    /// A ResNet-34 module with pre-trained weights.
-    #[cfg(feature = "pretrained")]
-    pub fn resnet34_pretrained(
-        weights: weights::ResNet34,
-        device: &Device<B>,
-    ) -> Result<Self, RecorderError> {
-        let weights = weights.weights();
-        let record = Self::load_weights_record(&weights, device)?;
-        let model = ResNet::<B>::resnet34(weights.num_classes, device).load_record(record);
-
-        Ok(model)
-    }
-
     /// ResNet-50 from [`Deep Residual Learning for Image Recognition`](https://arxiv.org/abs/1512.03385).
     ///
     /// # Arguments
@@ -151,29 +98,6 @@ impl<B: Backend> ResNet<B> {
         ResNetConfig::new(RESNET50_BLOCKS, num_classes, 4).init(device)
     }

-    /// ResNet-50 from [`Deep Residual Learning for Image Recognition`](https://arxiv.org/abs/1512.03385)
-    /// with pre-trained weights.
-    ///
-    /// # Arguments
-    ///
-    /// * `weights`: Pre-trained weights to load.
-    /// * `device` - Device to create the module on.
-    ///
-    /// # Returns
-    ///
-    /// A ResNet-50 module with pre-trained weights.
-    #[cfg(feature = "pretrained")]
-    pub fn resnet50_pretrained(
-        weights: weights::ResNet50,
-        device: &Device<B>,
-    ) -> Result<Self, RecorderError> {
-        let weights = weights.weights();
-        let record = Self::load_weights_record(&weights, device)?;
-        let model = ResNet::<B>::resnet50(weights.num_classes, device).load_record(record);
-
-        Ok(model)
-    }
-
     /// ResNet-101 from [`Deep Residual Learning for Image Recognition`](https://arxiv.org/abs/1512.03385).
     ///
     /// # Arguments
@@ -188,29 +112,6 @@ impl<B: Backend> ResNet<B> {
         ResNetConfig::new(RESNET101_BLOCKS, num_classes, 4).init(device)
     }

-    /// ResNet-101 from [`Deep Residual Learning for Image Recognition`](https://arxiv.org/abs/1512.03385)
-    /// with pre-trained weights.
-    ///
-    /// # Arguments
-    ///
-    /// * `weights`: Pre-trained weights to load.
-    /// * `device` - Device to create the module on.
-    ///
-    /// # Returns
-    ///
-    /// A ResNet-101 module with pre-trained weights.
-    #[cfg(feature = "pretrained")]
-    pub fn resnet101_pretrained(
-        weights: weights::ResNet101,
-        device: &Device<B>,
-    ) -> Result<Self, RecorderError> {
-        let weights = weights.weights();
-        let record = Self::load_weights_record(&weights, device)?;
-        let model = ResNet::<B>::resnet101(weights.num_classes, device).load_record(record);
-
-        Ok(model)
-    }
-
     /// ResNet-152 from [`Deep Residual Learning for Image Recognition`](https://arxiv.org/abs/1512.03385).
     ///
     /// # Arguments
@@ -225,29 +126,6 @@ impl<B: Backend> ResNet<B> {
         ResNetConfig::new(RESNET152_BLOCKS, num_classes, 4).init(device)
     }

-    /// ResNet-152 from [`Deep Residual Learning for Image Recognition`](https://arxiv.org/abs/1512.03385)
-    /// with pre-trained weights.
-    ///
-    /// # Arguments
-    ///
-    /// * `weights`: Pre-trained weights to load.
-    /// * `device` - Device to create the module on.
-    ///
-    /// # Returns
-    ///
-    /// A ResNet-152 module with pre-trained weights.
-    #[cfg(feature = "pretrained")]
-    pub fn resnet152_pretrained(
-        weights: weights::ResNet152,
-        device: &Device<B>,
-    ) -> Result<Self, RecorderError> {
-        let weights = weights.weights();
-        let record = Self::load_weights_record(&weights, device)?;
-        let model = ResNet::<B>::resnet152(weights.num_classes, device).load_record(record);
-
-        Ok(model)
-    }
-
     /// Re-initialize the last layer with the specified number of output classes.
     pub fn with_classes(mut self, num_classes: usize) -> Self {
         let [d_input, _d_output] = self.fc.weight.dims();
@@ -256,32 +134,6 @@ impl<B: Backend> ResNet<B> {
     }
 }

-#[cfg(feature = "pretrained")]
-impl<B: Backend> ResNet<B> {
-    /// Load specified pre-trained PyTorch weights as a record.
-    fn load_weights_record(
-        weights: &weights::Weights,
-        device: &Device<B>,
-    ) -> Result<ResNetRecord<B>, RecorderError> {
-        // Download torch weights
-        let torch_weights = weights.download().map_err(|err| {
-            RecorderError::Unknown(format!("Could not download weights.\nError: {err}"))
-        })?;
-
-        // Load weights from torch state_dict
-        let load_args = LoadArgs::new(torch_weights)
-            // Map *.downsample.0.* -> *.downsample.conv.*
-            .with_key_remap("(.+)\\.downsample\\.0\\.(.+)", "$1.downsample.conv.$2")
-            // Map *.downsample.1.* -> *.downsample.bn.*
-            .with_key_remap("(.+)\\.downsample\\.1\\.(.+)", "$1.downsample.bn.$2")
-            // Map layer[i].[j].* -> layer[i].blocks.[j].*
-            .with_key_remap("(layer[1-4])\\.([0-9]+)\\.(.+)", "$1.blocks.$2.$3");
-        let record = PyTorchFileRecorder::<FullPrecisionSettings>::new().load(load_args, device)?;
-
-        Ok(record)
-    }
-}
-
 /// [ResNet](ResNet) configuration.
 struct ResNetConfig {
     conv1: Conv2dConfig,
"#;

fn run<F>(name: &str, mut configure: F)
where
    F: FnMut(&mut Command) -> &mut Command,
{
    let mut command = Command::new(name);
    let configured = configure(&mut command);
    println!("Executing {:?}", configured);
    if !configured.status().unwrap().success() {
        panic!("failed to execute {:?}", configured);
    }
    println!("Command {:?} finished successfully", configured);
}

fn main() {
    let models_dir = std::env::temp_dir().join("models");
    let models_dir = models_dir.as_path();
    // Checkout ResNet code from models repo
    let models_dir = Path::new(models_dir);
    if !models_dir.join(".git").exists() {
        run("git", |command| {
            command
                .arg("clone")
                .arg("--depth=1")
                .arg("--no-checkout")
                .arg(MODELS_REPO)
                .arg(models_dir)
        });

        run("git", |command| {
            command
                .current_dir(models_dir)
                .arg("sparse-checkout")
                .arg("set")
                .arg("resnet-burn")
        });

        run("git", |command| {
            command.current_dir(models_dir).arg("checkout")
        });

        let patch_file = models_dir.join("benchmark.patch");

        fs::write(&patch_file, PATCH).expect("should write to file successfully");

        // Apply patch
        run("git", |command| {
            command
                .current_dir(models_dir)
                .arg("apply")
                .arg(patch_file.to_str().unwrap())
        });
    }

    // Copy contents to output dir
    let out_dir = env::var("OUT_DIR").unwrap();
    let source_path = models_dir.join("resnet-burn").join("resnet").join("src");
    let dest_path = Path::new(&out_dir);

    if let Ok(source_path) = fs::read_dir(source_path) {
        for file in source_path {
            let source_file = file.unwrap().path();
            let dest_file = dest_path.join(source_file.file_name().unwrap());
            fs::copy(source_file, dest_file).expect("should copy file successfully");
        }
    }

    // Delete cloned repository contents
    fs::remove_dir_all(models_dir.join(".git")).unwrap();
    fs::remove_dir_all(models_dir).unwrap();
}
