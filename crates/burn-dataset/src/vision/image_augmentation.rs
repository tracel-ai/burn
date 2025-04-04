use burn_tensor::{ElementConversion, Tensor, TensorData, backend::Backend};
use image::{ColorType, DynamicImage, RgbImage};
use rand::Rng;

/// Provides injectable traits for testing
pub trait InjectableTraits {
    /// Interface for swapping out a random generator function
    fn get_random<T>(&mut self) -> T
    where
        rand::distr::StandardUniform: rand::distr::Distribution<T>;
    /// An interface for swapping out a random generator function that takes a range
    fn get_random_with_range<R, T>(&mut self, range: R) -> T
    where
        R: rand::distr::uniform::SampleRange<T>,
        T: rand::distr::uniform::SampleUniform;
}

/// The injectable trait for the augmentation random number generator this is the default
/// rng generator struct used
pub struct RngDefault;

impl InjectableTraits for RngDefault {
    fn get_random<T>(&mut self) -> T
    where
        rand::distr::StandardUniform: rand::distr::Distribution<T>,
    {
        rand::rng().random::<T>()
    }

    fn get_random_with_range<R, T>(&mut self, range: R) -> T
    where
        R: rand::distr::uniform::SampleRange<T>,
        T: rand::distr::uniform::SampleUniform,
    {
        rand::random_range(range)
    }
}

/// Image augmentation and preprocessing utilities.
///
/// `Augmentations` provides randomized image transformations commonly used to
/// boost model performance through data augmentation.
///
/// # Supported Augmentations
///
/// - `random_photometric_distortion` — Adjusts brightness, contrast and hue with a set probability
/// - `random_zoom_out` — Pads and resizes to simulate zooming out with a set probability.
/// - `random_horizontal_flip` — Flips images horizontally with a set probability.
/// - `random_vertical_flip` — Flips images vertically with a set probability.
///
/// # Example
///
/// ```rust
/// let aug = Augmentations::new();
/// let output = aug.random_vertical_flip(image);
/// ```
///
/// # Notes
///
/// - Augmentations are applied probabilistically and may vary on each call.
/// - Designed to be composable and easily integrated into preprocessing pipelines.
/// - Aims to improve model robustness to lighting, scale, and spatial orientation.
#[derive(Clone, Debug)]
pub struct Augmentations<R: InjectableTraits = RngDefault> {
    rng: R,
}

/// Default provider for the random number generator used by the Augmentation implementation
impl Default for Augmentations {
    /// Initializes the Augmentation with a real (not seeded) random number generator
    fn default() -> Self {
        Self { rng: RngDefault }
    }
}

impl<R: InjectableTraits> Augmentations<R> {
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
    fn uniform_outcome_prob(&mut self, p: f32) -> bool {
        let max_val = ((1.0 / p.clamp(0.0, 1.0)) as u32) + 1;
        self.rng.get_random_with_range(1..max_val) == 1
    }
    /// Applies random photometric distortions to an RGB image, inspired by the
    /// data augmentation strategy used in SSD: Single Shot MultiBox Detector.
    ///
    /// Each transformation (brightness, contrast, hue) is applied independently
    /// with probability `p`, using values randomly sampled from the provided ranges.
    ///
    /// # Arguments
    ///
    /// * `orig_img` – The input RGB image to be distorted.
    /// * `brightness` – A `(min, max)` tuple specifying the range for brightness jitter.
    ///   The brightness factor is sampled uniformly from this range (non-negative).
    /// * `contrast` – A `(min, max)` tuple specifying the range for contrast jitter.
    ///   The contrast factor is sampled uniformly from this range (non-negative).
    /// * `hue` – A `(min, max)` tuple specifying the hue rotation range (0.0 ≤ min ≤ max ≤ 1.0).
    /// * `p` – The probability (between 0.0 and 1.0) that each individual operation is applied.
    ///
    /// # Returns
    ///
    /// A new `RgbImage` with photometric distortions applied probabilistically.
    /// If the operation is not applied (based on the probability), the original image is returned unchanged.
    pub fn random_photometeric_distort(
        &mut self,
        orig_img: RgbImage,
        brightness: (f32, f32),
        contrast: (f32, f32),
        hue: (f32, f32),
        p: f32,
    ) -> RgbImage {
        let mut img: DynamicImage = DynamicImage::from(orig_img);

        if Self::uniform_outcome_prob(self, p) {
            img = img.brighten(
                (self.rng.get_random_with_range(
                    brightness.0.clamp(0.0, 1.0)..brightness.1.clamp(0.0, 1.0),
                ) * 255.0) as i32,
            );
        }

        if Self::uniform_outcome_prob(self, p) {
            img = img.adjust_contrast(
                self.rng.get_random_with_range(
                    contrast.0.clamp(-1.0, 1.0)..contrast.1.clamp(-1.0, 1.0),
                ) * 100.0,
            );
        }

        if Self::uniform_outcome_prob(self, p) {
            img = img.huerotate(
                (self
                    .rng
                    .get_random_with_range(hue.0.clamp(0.0, 1.0)..hue.1.clamp(0.0, 1.0))
                    * 360.0) as i32,
            );
        }

        img.to_rgb8()
    }
    /// Applies a "zoom out" transformation by randomly padding the image, as described in
    /// the SSD: Single Shot MultiBox Detector paper.
    ///
    /// This augmentation simulates zooming out by increasing the canvas size and filling
    /// the surrounding space with a specified value. The new size is randomly selected
    /// within a given range, and the operation is applied with a specified probability.
    ///
    /// Some parts ported from:
    ///
    /// https://pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomZoomOut.html
    ///
    /// # Arguments
    ///
    /// * `orig_img` – The input RGB image to be padded.
    /// * `fill` – The pixel value used to fill the padded areas (e.g., 0 for black).
    /// * `side_range` – A `(min, max)` tuple specifying the range of scaling factors for the
    ///    output canvas size relative to the original image.
    /// * `p` – The probability (between 0.0 and 1.0) that the zoom-out transformation is applied.
    ///
    /// # Returns
    ///
    /// A new `RgbImage` with optional padding applied to simulate zooming out.
    /// If the operation is not applied (based on the probability), the original image is returned unchanged.
    ///
    /// # Notes
    ///
    /// - Padding is applied symmetrically and randomly positioned within the expanded canvas.
    /// - If the operation is not applied (based on `p`), the original image is returned unchanged.
    /// - This is useful for improving model robustness to scale and context variation.
    pub fn random_zoom_out<B: Backend>(
        &mut self,
        orig_img: RgbImage,
        fill: u8,
        side_range: (f32, f32),
        p: f32,
    ) -> RgbImage {
        if Self::uniform_outcome_prob(self, p) {
            let img_width = orig_img.width() as usize;
            let img_height = orig_img.height() as usize;

            if side_range.0 < 1.0 || side_range.0 > side_range.1 {
                panic!("Invalid side range provided {:#?}.", side_range);
            }

            let r = side_range.0 + self.rng.get_random::<f32>() * (side_range.1 - side_range.0);

            let canvas_width = (img_width as f32 * r) as usize;
            let canvas_height = (img_height as f32 * r) as usize;

            let r = (self.rng.get_random::<f32>(), self.rng.get_random::<f32>());

            let left = ((canvas_width - img_width) as f32 * r.0) as usize;
            let top = ((canvas_height - img_height) as f32 * r.1) as usize;
            let right = canvas_width - (left + img_width);
            let bottom = canvas_height - (top + img_height);

            // use a tensor to do the padding

            let ch_cnt: usize = ColorType::bytes_per_pixel(ColorType::Rgb8) as usize;

            // create a tensor large enough to hold the rgb image

            let t_shape: Vec<usize> = vec![1, img_height, img_width * ch_cnt];

            let device = B::Device::default();
            let tensor = Tensor::<B, 3>::from_data(
                TensorData::new(orig_img.into_vec(), t_shape).convert::<f32>(),
                &device,
            );

            // use channel count to pad the channels correctly in place

            let tensor = tensor.pad(
                (left * ch_cnt, right * ch_cnt, top, bottom),
                ElementConversion::elem::<f32>(fill as f32),
            );

            // convert tensor back to the new rgb image

            let to_rgb: Vec<u8> = tensor
                .to_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .map(|&p| p as u8)
                .collect();

            return RgbImage::from_vec(canvas_width as u32, canvas_height as u32, to_rgb).unwrap();
        }

        orig_img
    }

    /// Flips an RGB image horizontally with a given probability.
    ///
    /// This augmentation randomly flips the input image along the vertical axis (left ↔ right)
    /// with a probability `p`. It is commonly used to improve model robustness to horizontal
    /// variations in image orientation.
    ///
    /// # Arguments
    ///
    /// * `orig_img` – The input RGB image to flip.
    /// * `p` – The probability (between 0.0 and 1.0) that the horizontal flip will be applied.
    ///
    /// # Returns
    ///
    /// A new `RgbImage` that has been flipped horizontally if the operation was performed.
    /// If the operation is not applied (based on the probability), the original image is returned unchanged.
    pub fn random_horizontal_flip(&mut self, orig_img: RgbImage, p: f32) -> RgbImage {
        if Self::uniform_outcome_prob(self, p) {
            return horizontal_flip(orig_img);
        }

        orig_img
    }

    /// Flips an RGB image vertically with a given probability.
    ///
    /// This augmentation randomly flips the input image along the horizontal axis (top ↔ bottom)
    /// with a probability `p`. It is commonly used to improve model robustness to horizontal
    /// variations in image orientation.
    ///
    /// # Arguments
    ///
    /// * `orig_img` – The input RGB image to flip.
    /// * `p` – The probability (between 0.0 and 1.0) that the horizontal flip will be applied.
    ///
    /// # Returns
    ///
    /// A new `RgbImage` that has been flipped vertically if the operation was performed.
    /// If the operation is not applied (based on the probability), the original image is returned unchanged.
    pub fn random_vertical_flip(&mut self, orig_img: RgbImage, p: f32) -> RgbImage {
        if Self::uniform_outcome_prob(self, p) {
            return vertical_flip(orig_img);
        }

        orig_img
    }
}

// Currently uses image-rs in the future optimize by using internal tensor lib
fn horizontal_flip(orig_img: RgbImage) -> RgbImage {
    image::imageops::flip_horizontal(&orig_img)
}
// Currently uses image-rs in the future optimize by using internal tensor lib
fn vertical_flip(orig_img: RgbImage) -> RgbImage {
    image::imageops::flip_vertical(&orig_img)
}

#[cfg(test)]
mod tests {
    use std::hash::{DefaultHasher, Hash, Hasher};

    use super::*;
    use burn_wgpu::Wgpu;
    use rand::{SeedableRng, rngs::StdRng};

    pub fn create_test_image(width: u32, height: u32, pattern: [u8; 3]) -> RgbImage {
        let mut img = RgbImage::new(width, height);
        let img_pattern: image::Rgb<u8> = image::Rgb(pattern);

        for px in img.pixels_mut() {
            *px = img_pattern;
        }

        img
    }

    struct SeededRng {
        rng: StdRng,
    }

    impl SeededRng {
        fn new(seed: u64) -> Self {
            let seeded_rng = StdRng::seed_from_u64(seed);
            Self { rng: seeded_rng }
        }
    }

    impl InjectableTraits for SeededRng {
        fn get_random<T>(&mut self) -> T
        where
            rand::distr::StandardUniform: rand::distr::Distribution<T>,
        {
            self.rng.random::<T>()
        }

        fn get_random_with_range<R, T>(&mut self, range: R) -> T
        where
            R: rand::distr::uniform::SampleRange<T>,
            T: rand::distr::uniform::SampleUniform,
        {
            self.rng.random_range(range)
        }
    }

    #[test]
    fn test_seeded_random_number_generation() {
        let mut mse = SeededRng::new(3);
        let mut test_vec = Vec::<i32>::new();
        let expected_vec = vec![-1513825812, 408920382, -83330236, 1513922966, 612228279];

        for _ in 0..5 {
            test_vec.push(mse.get_random::<i32>());
        }

        let mut h = DefaultHasher::new();
        test_vec.hash(&mut h);
        assert_eq!(expected_vec, test_vec);
    }

    #[test]
    fn random_zoom_test() {
        let mse = SeededRng::new(3);
        let img = create_test_image(12, 12, [1, 2, 3]);
        let test_success_hash: u64 = 18124238768780137715;

        type ZoomTestBackend = Wgpu<f32, i32>;

        let mut aug = Augmentations::new(mse);

        let img = aug.random_zoom_out::<ZoomTestBackend>(img, 0, (1.0, 4.0), 1.0);

        let mut h = DefaultHasher::new();
        img.hash(&mut h);
        assert_eq!(test_success_hash, h.finish());
    }

    #[test]
    fn random_photometeric_test() {
        let mse = SeededRng::new(3);
        let img = create_test_image(12, 12, [128, 128, 255]);
        let test_success_hash: u64 = 3249002746881015726;

        let mut aug = Augmentations::new(mse);

        let img = aug.random_photometeric_distort(img, (0.0, 0.3), (-0.5, 0.5), (0.0, 1.0), 0.5);

        let mut h = DefaultHasher::new();
        img.hash(&mut h);
        assert_eq!(test_success_hash, h.finish());
    }

    #[test]
    fn random_horizontal_flip_test() {
        let mse = SeededRng::new(3);
        let img = create_test_image(12, 12, [1, 2, 3]);
        let test_success_hash: u64 = 8226555038773849619;

        type FlipTestBackend = Wgpu<f32, i32>;

        let mut aug = Augmentations::new(mse);

        let img = aug.random_zoom_out::<FlipTestBackend>(img, 0, (1.0, 10.0), 1.0);

        let img = aug.random_horizontal_flip(img, 1.0);

        let mut h = DefaultHasher::new();
        img.hash(&mut h);
        assert_eq!(test_success_hash, h.finish());
    }

    #[test]
    fn random_vertical_flip_test() {
        let mse = SeededRng::new(3);
        let img = create_test_image(12, 12, [1, 2, 3]);
        let test_success_hash: u64 = 16449163701239239957;

        type MyBackend = Wgpu<f32, i32>;

        let mut aug = Augmentations::new(mse);

        let img = aug.random_zoom_out::<MyBackend>(img, 0, (1.0, 10.0), 1.0);

        let img = aug.random_vertical_flip(img, 1.0);

        let mut h = DefaultHasher::new();
        img.hash(&mut h);
        assert_eq!(test_success_hash, h.finish());
    }
}
