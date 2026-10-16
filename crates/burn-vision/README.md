# Burn Vision

> Computer vision operations for [Burn](https://github.com/tracel-ai/burn), with GPU acceleration where possible

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-vision.svg)](https://crates.io/crates/burn-vision)
[![Documentation](https://docs.rs/burn-vision/badge.svg)](https://docs.rs/burn-vision)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT)

Operation names follow OpenCV wherever applicable:

- `connected_components` and `connected_components_with_stats`
- morphology: `dilate` and `erode`, with `create_structuring_element`
- `nms` (non-maximum suppression)
- `filter2d` (depthwise 2D correlation)
- color conversion: `rgb2gray`, `gray2rgb`, `rgb2hsv` and `hsv2rgb`
- 2D affine transforms with `Transform2D`: rotation, scale, shear and translation
- with the `loss` feature, a VGG19-based Gram matrix (style) loss

## Usage

Enable both `vision` and a backend feature on `burn`, and use the operations through
`burn::vision`:

```toml
burn = { version = "0.22", features = ["vision", "flex"] }
```

No execution backend is enabled by default when depending on this crate directly; select `flex`,
`wgpu` or another backend feature on it. Enabling only `burn/flex` does not enable this crate's
backend implementations.

<!-- burn-crate-footer -->

---

Part of the [Burn](https://github.com/tracel-ai/burn) deep learning framework. See the
[Burn Book](https://burn.dev/books/burn/) and the [API documentation](https://docs.rs/burn).
