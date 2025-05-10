use super::image_ops;
use burn_tensor::{ElementConversion, Tensor, backend::Backend};
use rand::{SeedableRng, rngs::StdRng};

/// Image augmentation and preprocessing utilities.
///
/// `Augmentations` provides randomized image transformations commonly used to
/// boost model performance through data augmentation.
///
/// # Supported Augmentations
///
/// - `random_photometric_distortion` — Adjusts brightness, contrast and hue with a set probability
/// - `random_zoom_out` — Pads and resizes to simulate zooming out with a set probability.
/// - `random_vertical_flip` — Flips images vertically with a set probability.
///
/// # Notes
///
/// - Augmentations are applied probabilistically and may vary on each call.
/// - Aims to improve model robustness to lighting, scale, and spatial orientation.
#[derive(Clone, Debug)]
pub struct Augmentations<R>
where
    R: rand::Rng,
{
    rng: R,
}

/// Provides a default implementation for `Augmentations`.
///
/// This creates a ready-to-use `Augmentations` instance with standard settings.
impl Default for Augmentations<StdRng> {
    /// Creates a default `Augmentations` instance.
    ///
    /// This is the recommended method for most users to obtain a ready-to-use
    /// augmentation with standard behavior.
    ///
    /// # Returns
    ///
    /// A new `Augmentations` instance with default settings.
    fn default() -> Self {
        Self {
            rng: StdRng::from_os_rng(),
        }
    }
}

impl<R: rand::Rng> Augmentations<R> {
    /// Creates a new instance of `Augmentations` with the given random number generator.
    ///
    /// # Arguments
    ///
    /// * `rng` – The random number generator to be used for augmentations.
    ///
    /// # Returns
    ///
    /// A new instance of `Augmentations` with the provided RNG.
    pub fn new(rng: R) -> Self {
        Self { rng }
    }

    /// Returns a boolean result based on a uniform random probability.
    ///
    /// This function generates a random boolean value, where the probability of returning
    /// `true` is determined by `p`.
    ///
    /// # Arguments
    ///
    /// * `p` – The probability (between 0.0 and 1.0) that the function will return `true`.
    ///
    /// # Returns
    ///
    /// A boolean value: `true` with probability `p`.
    ///
    /// # Remarks
    ///
    /// - If `p` is 0.0, the function will always return `false`.
    /// - If `p` is 1.0, the function will always return `true`.
    /// - Values outside the range [0.0, 1.0] are clamped to this range.
    fn should_apply(&mut self, p: f32) -> bool {
        self.rng.random::<f32>() < p.clamp(0.0, 1.0)
    }

    /// Applies random photometric distortions to a 3-channel image tensor, inspired by the
    /// data augmentation strategy used in SSD: Single Shot MultiBox Detector.
    ///
    /// Each transformation (brightness, contrast, hue) is applied independently
    /// with probability `p`, using values randomly sampled from the specified ranges.
    ///
    /// # Arguments
    ///
    /// * `img_tnsr` – A 3D tensor representing the image, in shape `[3, H, W]` (channel-first format).
    /// * `brightness` – A `(min, max)` tuple specifying the brightness adjustment range.
    ///   The brightness factor is sampled uniformly from this range (`-1.0 ≤ min ≤ max ≤ 1.0`).
    /// * `contrast` – A `(min, max)` tuple specifying the contrast adjustment range (as a percentage).
    ///   The contrast factor is sampled uniformly from this range (`-1.0 ≤ min ≤ max ≤ 1.0`).
    /// * `hue` – A `(min, max)` tuple specifying the hue rotation range in degrees
    ///   (`-180.0 ≤ min ≤ max ≤ 180.0`).
    /// * `p` – The probability (between 0.0 and 1.0) that each individual transformation is applied.
    ///
    /// # Returns
    ///
    /// A new image tensor of shape `[3, H, W]` with photometric distortions applied probabilistically.
    /// If a transformation is not applied (based on probability `p`), the corresponding attribute remains unchanged.
    pub fn random_photometric_distort<B: Backend>(
        &mut self,
        mut img_tnsr: Tensor<B, 3>,
        brightness: (f32, f32),
        contrast: (f32, f32),
        hue: (f32, f32),
        p: f32,
    ) -> Tensor<B, 3> {
        if self.should_apply(p) {
            let r_bright = (self
                .rng
                .random_range(brightness.0.clamp(-1.0, 1.0)..brightness.1.clamp(-1.0, 1.0))
                * image_ops::MAX_PIXEL_VAL) as i32;
            img_tnsr = image_ops::brighten(img_tnsr, r_bright);
        }

        if self.should_apply(p) {
            let r_contrast = self
                .rng
                .random_range(contrast.0.clamp(-1.0, 1.0)..contrast.1.clamp(-1.0, 1.0))
                * 100.0;
            img_tnsr = image_ops::contrast(img_tnsr, r_contrast);
        }

        if self.should_apply(p) {
            let r_hue_rot = self
                .rng
                .random_range(hue.0.clamp(-1.0, 1.0)..hue.1.clamp(-1.0, 1.0) * 180.0);

            img_tnsr = image_ops::hue_rotate(img_tnsr, r_hue_rot);
        }

        img_tnsr
    }

    /// Applies a "zoom out" transformation by randomly padding the image and adjusting bounding boxes, as described in
    /// the SSD: Single Shot MultiBox Detector paper.
    ///
    /// This augmentation simulates zooming out by increasing the canvas size and filling
    /// the surrounding space with a specified value. The new canvas size is randomly selected
    /// within a given range, and the transformation is applied with a specified probability.
    ///
    /// Some parts inspired by:
    /// https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomZoomOut.html
    ///
    /// # Arguments
    ///
    /// * `img_data` – A tuple containing:
    ///     - A 3D tensor representing the image, in shape `[3, H, W]` (channel-first format).
    ///     - An optional vector of bounding box tensors, each of shape `[4]` in `[x, y, w, h]` format.
    ///       If provided, the bounding boxes are adjusted to match the new padded image dimensions.
    /// * `fill` – The pixel value used to fill the padded areas (e.g., `0.0` for black).
    /// * `side_range` – A `(min, max)` tuple specifying the range of scaling factors for the
    ///   output canvas size relative to the original image.
    /// * `p` – The probability (between 0.0 and 1.0) that the zoom-out transformation is applied.
    ///
    /// # Returns
    ///
    /// Returns a new tuple:
    /// - The padded (zoomed-out) image tensor.
    /// - The adjusted bounding box tensors (if provided), updated to match the new image size.
    ///   If the operation is not applied (based on the probability), the original image and bounding boxes
    ///   are returned unchanged.
    pub fn random_zoom_out<B: Backend>(
        &mut self,
        img_data: (Tensor<B, 3>, Option<Vec<Tensor<B, 1>>>),
        fill: u8,
        side_range: (f32, f32),
        p: f32,
    ) -> (Tensor<B, 3>, Option<Vec<Tensor<B, 1>>>) {
        if !self.should_apply(p) {
            return img_data;
        }

        let (cur_img_tnsr, mut maybe_bb_tnsr_list) = img_data;
        let [_ch, height, width] = cur_img_tnsr.dims();

        if side_range.0 < 1.0 || side_range.0 > side_range.1 {
            panic!("Invalid side range provided {:#?}.", side_range);
        }

        let r = side_range.0 + self.rng.random::<f32>() * (side_range.1 - side_range.0);

        let canvas_width = (width as f32 * r) as usize;
        let canvas_height = (height as f32 * r) as usize;

        let r = (self.rng.random::<f32>(), self.rng.random::<f32>());

        let left = ((canvas_width - width) as f32 * r.0) as usize;
        let top = ((canvas_height - height) as f32 * r.1) as usize;
        let right = canvas_width - (left + width);
        let bottom = canvas_height - (top + height);

        // Pad image
        let new_img_tnsr = cur_img_tnsr.pad(
            (left, right, top, bottom),
            ElementConversion::elem::<f32>(fill as f32),
        );

        // Translate bounding box
        if let Some(cur_bb_t_list) = maybe_bb_tnsr_list.as_mut() {
            let device = new_img_tnsr.device();
            let mut tmp_bb_tnsr_list = Vec::<Tensor<B, 1>>::new();
            for bbox in cur_bb_t_list.iter() {
                let trans = Tensor::<B, 1>::from_data([left as f32, top as f32, 0.0, 0.0], &device);
                tmp_bb_tnsr_list.push(bbox.clone().add(trans));
            }

            maybe_bb_tnsr_list = Some(tmp_bb_tnsr_list);
        }

        (new_img_tnsr, maybe_bb_tnsr_list)
    }

    /// Flips a 3-channel image tensor of shape `[C, H, W]` and its associated bounding boxes vertically
    /// with a given probability.
    ///
    /// This augmentation randomly flips the input image along the horizontal axis (top ↔ bottom) and
    /// its associated bounding boxes with a probability `p`.
    ///
    /// # Arguments
    ///
    /// * `img_data` – A tuple containing:
    ///     - A 3D tensor representing the image, in shape `[3, H, W]` (channel-first format).
    ///     - An optional vector of bounding box tensors, each of shape `[4]` in `[x, y, w, h]` format,
    ///       representing object locations in the image. If provided, the bounding boxes will also be flipped
    ///       vertically to remain aligned with the transformed image.
    /// * `p` – The probability (between 0.0 and 1.0) that the vertical flip will be applied.
    ///
    /// # Returns
    ///
    /// Returns a new tuple:
    /// - The vertically flipped image tensor.
    /// - The vertically flipped bounding box tensors (if provided), adjusted for the new vertical orientation,
    ///   still in `[x, y, w, h]` format.
    ///   If the operation is not applied (based on the probability), the original image and bounding boxes
    ///   are returned unchanged.
    pub fn random_vertical_flip<B: Backend>(
        &mut self,
        img_data: (Tensor<B, 3>, Option<Vec<Tensor<B, 1>>>),
        p: f32,
    ) -> (Tensor<B, 3>, Option<Vec<Tensor<B, 1>>>) {
        if !self.should_apply(p) {
            return img_data;
        }
        image_ops::vertical_flip(img_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::BoundingBox;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use rand::Rng;
    use std::hash::{DefaultHasher, Hash, Hasher};

    #[test]
    fn test_seeded_random_number_generation() {
        let mut mse = StdRng::seed_from_u64(3);
        let mut test_vec = Vec::<i32>::new();
        let expected_vec = vec![-1513825812, 408920382, -83330236, 1513922966, 612228279];

        for _ in 0..5 {
            test_vec.push(mse.random::<i32>());
        }

        assert_eq!(expected_vec, test_vec);
    }

    #[test]
    fn random_zoom_test() {
        let img = image_ops::create_test_image(32, 32, [1, 2, 3]);

        let mut bb_list = Vec::<Tensor<NdArray<f32>, 1>>::new();
        let device = NdArrayDevice::default();

        let bb = BoundingBox {
            coords: [3.0, 4.0, 2.0, 2.0],
            label: 0,
        };

        bb_list.push(image_ops::bbox_as_tensor::<NdArray<f32>>(bb, &device));

        let bb = BoundingBox {
            coords: [1.0, 2.0, 3.0, 4.0],
            label: 0,
        };

        bb_list.push(image_ops::bbox_as_tensor::<NdArray<f32>>(bb, &device));

        let mut aug = Augmentations::new(StdRng::seed_from_u64(3));

        let image_t = image_ops::rgb_img_as_tensor::<NdArray>(img, &device);

        let (img_tensor, bb_tlist) =
            aug.random_zoom_out::<NdArray>((image_t, Some(bb_list)), 0, (1.0, 4.0), 1.0);

        let (img_tensor, bb_tlist) = aug.random_vertical_flip((img_tensor, bb_tlist), 1.0);

        // Check image test result
        let test_success_hash: u64 = 1697389267139162335;
        let mut h = DefaultHasher::new();
        img_tensor.into_data().as_bytes().hash(&mut h);
        assert_eq!(test_success_hash, h.finish());

        // Check bounding box translations
        let bb_vec = bb_tlist.unwrap();

        let eq_test_t = bb_vec[0].to_data();
        let eq_test_t: Vec<i32> = eq_test_t
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|&x| x as i32)
            .collect();

        assert_eq!(eq_test_t.as_slice(), [11, 32, 2, 2]);

        let eq_test_t = bb_vec[1].to_data();
        let eq_test_t: Vec<i32> = eq_test_t
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|&x| x as i32)
            .collect();

        assert_eq!(eq_test_t.as_slice(), [9, 32, 3, 4]);
    }

    #[test]
    fn random_photometeric_test() {
        let img = image_ops::create_test_image(12, 12, [128, 128, 255]);
        let test_success_hash: u64 = 13135064361691146484;
        let device = NdArrayDevice::default();

        let image_t = image_ops::rgb_img_as_tensor::<NdArray<f32>>(img, &device);
        let mut aug = Augmentations::new(StdRng::seed_from_u64(3));

        let image_t =
            aug.random_photometric_distort(image_t, (0.0, 0.3), (-0.5, 0.5), (0.0, 1.0), 0.5);

        // Test hash of image
        let mut h = DefaultHasher::new();
        image_t.to_data().as_bytes().hash(&mut h);
        assert_eq!(test_success_hash, h.finish());
    }
}
