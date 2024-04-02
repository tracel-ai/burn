use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

const MODELS_DIR: &str = "/tmp/models";
const MODELS_REPO: &str = "https://github.com/tracel-ai/models.git";
// Patch code
// 1) ReLU -> Relu
// 2) disable `init_with` methods to suppress unused warnings
const PATCH: &str = r#"diff --git a/resnet-burn/src/model/block.rs b/resnet-burn/src/model/block.rs
index 7d92554..408c51a 100644
--- a/resnet-burn/src/model/block.rs
+++ b/resnet-burn/src/model/block.rs
@@ -7,7 +7,7 @@ use burn::{
     module::Module,
     nn::{
         conv::{Conv2d, Conv2dConfig},
-        BatchNorm, BatchNormConfig, Initializer, PaddingConfig2d, ReLU,
+        BatchNorm, BatchNormConfig, Initializer, PaddingConfig2d, Relu,
     },
     tensor::{backend::Backend, Device, Tensor},
 };
@@ -22,7 +22,7 @@ pub trait ResidualBlock<B: Backend> {
 pub struct BasicBlock<B: Backend> {
     conv1: Conv2d<B>,
     bn1: BatchNorm<B, 2>,
-    relu: ReLU,
+    relu: Relu,
     conv2: Conv2d<B>,
     bn2: BatchNorm<B, 2>,
     downsample: Option<Downsample<B>>,
@@ -63,7 +63,7 @@ impl<B: Backend> ResidualBlock<B> for BasicBlock<B> {
 pub struct Bottleneck<B: Backend> {
     conv1: Conv2d<B>,
     bn1: BatchNorm<B, 2>,
-    relu: ReLU,
+    relu: Relu,
     conv2: Conv2d<B>,
     bn2: BatchNorm<B, 2>,
     conv3: Conv2d<B>,
@@ -187,7 +187,7 @@ impl BasicBlockConfig {
                 .with_initializer(initializer.clone())
                 .init(device),
             bn1: self.bn1.init(device),
-            relu: ReLU::new(),
+            relu: Relu::new(),
             conv2: self
                 .conv2
                 .clone()
@@ -199,11 +199,12 @@ impl BasicBlockConfig {
     }
 
     /// Initialize a new [basic residual block](BasicBlock) module with a [record](BasicBlockRecord).
+    #[cfg(feature = "pretrained")]
     fn init_with<B: Backend>(&self, record: BasicBlockRecord<B>) -> BasicBlock<B> {
         BasicBlock {
             conv1: self.conv1.init_with(record.conv1),
             bn1: self.bn1.init_with(record.bn1),
-            relu: ReLU::new(),
+            relu: Relu::new(),
             conv2: self.conv2.init_with(record.conv2),
             bn2: self.bn2.init_with(record.bn2),
             downsample: self.downsample.as_ref().map(|d| {
@@ -286,7 +287,7 @@ impl BottleneckConfig {
                 .with_initializer(initializer.clone())
                 .init(device),
             bn1: self.bn1.init(device),
-            relu: ReLU::new(),
+            relu: Relu::new(),
             conv2: self
                 .conv2
                 .clone()
@@ -304,11 +305,12 @@ impl BottleneckConfig {
     }
 
     /// Initialize a new [bottleneck residual block](Bottleneck) module with a [record](BottleneckRecord).
+    #[cfg(feature = "pretrained")]
     fn init_with<B: Backend>(&self, record: BottleneckRecord<B>) -> Bottleneck<B> {
         Bottleneck {
             conv1: self.conv1.init_with(record.conv1),
             bn1: self.bn1.init_with(record.bn1),
-            relu: ReLU::new(),
+            relu: Relu::new(),
             conv2: self.conv2.init_with(record.conv2),
             bn2: self.bn2.init_with(record.bn2),
             conv3: self.conv3.init_with(record.conv3),
@@ -358,6 +360,7 @@ impl DownsampleConfig {
     }
 
     /// Initialize a new [downsample](Downsample) module with a [record](DownsampleRecord).
+    #[cfg(feature = "pretrained")]
     fn init_with<B: Backend>(&self, record: DownsampleRecord<B>) -> Downsample<B> {
         Downsample {
             conv: self.conv.init_with(record.conv),
@@ -412,6 +415,7 @@ impl<B: Backend> LayerBlockConfig<BasicBlock<B>> {
 
     /// Initialize a new [LayerBlock](LayerBlock) module with a [record](LayerBlockRecord) for
     /// [basic residual blocks](BasicBlock).
+    #[cfg(feature = "pretrained")]
     pub fn init_with(
         &self,
         record: LayerBlockRecord<B, BasicBlock<B>>,
@@ -461,6 +465,7 @@ impl<B: Backend> LayerBlockConfig<Bottleneck<B>> {
 
     /// Initialize a new [LayerBlock](LayerBlock) module with a [record](LayerBlockRecord) for
     /// [bottleneck residual blocks](Bottleneck).
+    #[cfg(feature = "pretrained")]
     pub fn init_with(
         &self,
         record: LayerBlockRecord<B, Bottleneck<B>>,
diff --git a/resnet-burn/src/model/resnet.rs b/resnet-burn/src/model/resnet.rs
index 0bc6a6c..2504159 100644
--- a/resnet-burn/src/model/resnet.rs
+++ b/resnet-burn/src/model/resnet.rs
@@ -6,12 +6,13 @@ use burn::{
     nn::{
         conv::{Conv2d, Conv2dConfig},
         pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
-        BatchNorm, BatchNormConfig, Initializer, Linear, LinearConfig, PaddingConfig2d, ReLU,
+        BatchNorm, BatchNormConfig, Initializer, Linear, LinearConfig, PaddingConfig2d, Relu,
     },
     tensor::{backend::Backend, Device, Tensor},
 };
 
-use super::block::{BasicBlock, Bottleneck, LayerBlock, LayerBlockConfig, ResidualBlock};
+pub use super::block::{BasicBlock, Bottleneck};
+use super::block::{LayerBlock, LayerBlockConfig, ResidualBlock};
 
 #[cfg(feature = "pretrained")]
 use {
@@ -33,7 +34,7 @@ const RESNET152_BLOCKS: [usize; 4] = [3, 8, 36, 3];
 pub struct ResNet<B: Backend, M> {
     conv1: Conv2d<B>,
     bn1: BatchNorm<B, 2>,
-    relu: ReLU,
+    relu: Relu,
     maxpool: MaxPool2d,
     layer1: LayerBlock<B, M>,
     layer2: LayerBlock<B, M>,
@@ -360,7 +361,7 @@ impl<B: Backend> ResNetConfig<B, BasicBlock<B>> {
         ResNet {
             conv1: self.conv1.with_initializer(initializer).init(device),
             bn1: self.bn1.init(device),
-            relu: ReLU::new(),
+            relu: Relu::new(),
             maxpool: self.maxpool.init(),
             layer1: self.layer1.init(device),
             layer2: self.layer2.init(device),
@@ -372,11 +373,12 @@ impl<B: Backend> ResNetConfig<B, BasicBlock<B>> {
     }
 
     /// Initialize a new [ResNet](ResNet) module with a [record](ResNetRecord).
+    #[cfg(feature = "pretrained")]
     fn init_with(&self, record: ResNetRecord<B, BasicBlock<B>>) -> ResNet<B, BasicBlock<B>> {
         ResNet {
             conv1: self.conv1.init_with(record.conv1),
             bn1: self.bn1.init_with(record.bn1),
-            relu: ReLU::new(),
+            relu: Relu::new(),
             maxpool: self.maxpool.init(),
             layer1: self.layer1.init_with(record.layer1),
             layer2: self.layer2.init_with(record.layer2),
@@ -400,7 +402,7 @@ impl<B: Backend> ResNetConfig<B, Bottleneck<B>> {
         ResNet {
             conv1: self.conv1.with_initializer(initializer).init(device),
             bn1: self.bn1.init(device),
-            relu: ReLU::new(),
+            relu: Relu::new(),
             maxpool: self.maxpool.init(),
             layer1: self.layer1.init(device),
             layer2: self.layer2.init(device),
@@ -412,11 +414,12 @@ impl<B: Backend> ResNetConfig<B, Bottleneck<B>> {
     }
 
     /// Initialize a new [ResNet](ResNet) module with a [record](ResNetRecord).
+    #[cfg(feature = "pretrained")]
     fn init_with(&self, record: ResNetRecord<B, Bottleneck<B>>) -> ResNet<B, Bottleneck<B>> {
         ResNet {
             conv1: self.conv1.init_with(record.conv1),
             bn1: self.bn1.init_with(record.bn1),
-            relu: ReLU::new(),
+            relu: Relu::new(),
             maxpool: self.maxpool.init(),
             layer1: self.layer1.init_with(record.layer1),
             layer2: self.layer2.init_with(record.layer2),
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
    // Checkout ResNet code from models repo
    let models_dir = Path::new(MODELS_DIR);
    if !models_dir.join(".git").exists() {
        run("git", |command| {
            command
                .arg("clone")
                .arg("--depth=1")
                .arg("--no-checkout")
                .arg(MODELS_REPO)
                .arg(MODELS_DIR)
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

        // TODO: remove Relu patch when the models dir is updated to use Burn v0.13.0
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
    let source_path = models_dir.join("resnet-burn").join("src").join("model");
    let dest_path = Path::new(&out_dir);

    for file in fs::read_dir(source_path).unwrap() {
        let source_file = file.unwrap().path();
        let dest_file = dest_path.join(source_file.file_name().unwrap());
        fs::copy(source_file, dest_file).expect("should copy file successfully");
    }

    // Delete cloned repository contents
    fs::remove_dir_all(models_dir.join(".git")).unwrap();
    fs::remove_dir_all(models_dir).unwrap();
}
