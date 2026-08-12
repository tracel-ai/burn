# ResNet integration-test fixture

The ResNet-18 inference fixture in `resnet.rs` is adapted from the
`resnet-burn` model in the Tracel AI models repository and from torchvision's
ResNet implementation. It is included under Burn's MIT OR Apache-2.0 license.

The fixture intentionally excludes pretrained weights, download code, training
code, and model variants that are not exercised by the ONNX exporter test.
