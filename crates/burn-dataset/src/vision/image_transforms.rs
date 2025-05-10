use super::{BoundingBox, ImageDatasetItem, image_folder, image_ops};
use crate::vision::PixelDepth;
use burn_tensor::backend::{self, Backend};
use burn_tensor::{Shape, Tensor, TensorData};
use image_folder::Annotation;
use std::io::Error;

#[derive(Debug, Clone)]
pub enum AnnotationTnsr<B: Backend> {
    /// Image-level label.
    Label(usize),
    /// Multiple image-level labels.
    MultiLabel(Vec<usize>),
    /// Object bounding boxes.
    BoundingBoxes(Vec<Tensor<B, 1>>),
    /// Segmentation mask.
    SegmentationMask,
} // BM&JJ

/// Converts a bounding box into a tensor.
///
/// Takes a `BoundingBox` and converts it into a 1D tensor.
///
/// # Parameters
/// - `bbox`: The bounding box to be converted.
///
/// # Returns
/// An `Option` containing the `[x, y, w, h]` tensor representing the bounding box,
/// or `None` if the bounding box is invalid.
pub fn bbox_as_tensor<B: Backend>(bbox: &BoundingBox) -> Tensor<B, 1> {
    let device = B::Device::default();
    Tensor::<B, 1>::from_data(bbox.coords, &device)
}

fn image_as_vec_u8(image: Vec<PixelDepth>) -> Vec<u8> {
    image
        .into_iter()
        .map(|p: PixelDepth| -> u8 { p.try_into().unwrap() })
        .collect::<Vec<u8>>()
}

/// todo
#[derive(Debug, Clone)]
pub struct Sample<B: Backend> {
    image: Tensor<B, 3>,
    annotations: Vec<AnnotationTnsr<B>>,
}

/// doc
pub trait IntoSample {
    fn into_sample<B: Backend>(self) -> Sample<B>;
}

impl IntoSample for ImageDatasetItem {
    fn into_sample<B: Backend>(self) -> Sample<B> {
        let device = B::Device::default();

        let tnsr_data = TensorData::new(
            image_as_vec_u8(self.image),
            Shape::new([self.image_dims.0 as usize, self.image_dims.1 as usize, 3]),
        );

        let mut sample = Sample {
            image: Tensor::<B, 3>::from_data(tnsr_data.convert::<B::FloatElem>(), &device)
                .permute([2, 0, 1]),
            annotations: vec![],
        };

        // ImageDatasetItem can only have one type of annotation at a time as implemented now
        sample.annotations.push(match &self.annotation {
            Annotation::BoundingBoxes(bboxes) => {
                let bboxes_tnsr_lst = bboxes
                    .iter()
                    .map(|bbox| bbox_as_tensor::<B>(bbox))
                    .collect();
                AnnotationTnsr::BoundingBoxes(bboxes_tnsr_lst)
            }
            Annotation::Label(_) => todo!(),
            Annotation::MultiLabel(_) => todo!(),
            Annotation::SegmentationMask(_) => todo!(),
        });

        sample
    }
}

/// Todo
pub trait Transform {
    /// Todo
    fn apply_image<B: Backend>(&self, sample: Sample<B>) -> Result<Sample<B>, Error>;
    /// Todo
    fn apply_annotations<B: Backend>(&self, sample: Sample<B>) -> Result<Sample<B>, Error> {
        // Default no-op. Should be implemented for transformations that impact the annotations.
        Ok(sample)
    }
    /// Todo
    fn apply<B: Backend>(&self, sample: Sample<B>) -> Result<Sample<B>, Error> {
        let sample = self.apply_image(sample)?;
        self.apply_annotations(sample)
    }
}

/// Todo
pub struct VerticalFlip {}

/// Todo
pub fn bbox_vertical_flip<B: Backend>(
    img_tnsr: &Tensor<B, 3>,
    bboxes: &[Tensor<B, 1>],
) -> Vec<Tensor<B, 1>> {
    let [_ch, height, _width] = img_tnsr.dims();

    // Flip bounding boxes vertically

    let device = img_tnsr.device();
    let mut trans_bb_tnsr_list = Vec::<Tensor<B, 1>>::new();
    for bbox_tnsr in bboxes.iter() {
        let mut trans = bbox_tnsr.clone().into_data().to_vec::<f32>().unwrap();
        trans[1] = height as f32 - trans[1] - trans[3];
        trans_bb_tnsr_list.push(Tensor::<B, 1>::from_data(
            TensorData::new(trans, [4]),
            &device,
        ));
    }

    trans_bb_tnsr_list
}

impl Transform for VerticalFlip {
    fn apply_image<B: Backend>(&self, mut sample: Sample<B>) -> Result<Sample<B>, Error> {
        sample.image = image_ops::vertical_flip(sample.image);
        Ok(sample)
    }

    fn apply_annotations<B: Backend>(&self, mut sample: Sample<B>) -> Result<Sample<B>, Error> {
        for anno in sample.annotations.iter_mut() {
            *anno = match anno {
                AnnotationTnsr::BoundingBoxes(bboxes) => {
                    AnnotationTnsr::BoundingBoxes(bbox_vertical_flip(&sample.image, bboxes))
                }
                AnnotationTnsr::Label(_) => todo!(),
                AnnotationTnsr::MultiLabel(_) => todo!(),
                AnnotationTnsr::SegmentationMask => todo!(),
            }
        }

        Ok(sample)
    }
}

/// doc
pub struct Zoom {
    fill: u8,
    top: usize,
    bottom: usize,
    right: usize,
    left: usize,
}

/// doc
pub fn bboxes_zoom_out<B: Backend>(
    bboxes: &[Tensor<B, 1>],
    left: usize,
    top: usize,
) -> Vec<Tensor<B, 1>> {
    let device = bboxes[0].device();
    let mut bb_tnsr_list = Vec::<Tensor<B, 1>>::new();
    for bbox in bboxes.iter() {
        let trans = Tensor::<B, 1>::from_data([left as f32, top as f32, 0.0, 0.0], &device);
        bb_tnsr_list.push(bbox.clone().add(trans));
    }

    bb_tnsr_list
}

impl Transform for Zoom {
    fn apply_image<B: Backend>(&self, mut sample: Sample<B>) -> Result<Sample<B>, Error> {

        
        sample.image = image_ops::zoom_out(
            sample.image,
            self.left,
            self.right,
            self.top,
            self.bottom,
            self.fill,
        );

        Ok(sample)
    }

    fn apply_annotations<B: Backend>(&self, mut sample: Sample<B>) -> Result<Sample<B>, Error> {
        for anno in sample.annotations.iter_mut() {
            *anno = match anno {
                AnnotationTnsr::BoundingBoxes(bboxes) => {
                    AnnotationTnsr::BoundingBoxes(bboxes_zoom_out(bboxes, self.left, self.top))
                }
                AnnotationTnsr::Label(_) => todo!(),
                AnnotationTnsr::MultiLabel(_) => todo!(),
                AnnotationTnsr::SegmentationMask => todo!(),
            }
        }

        Ok(sample)
    }
}
pub struct Contrast {
    value: f32,
}

impl Transform for Contrast {
    fn apply_image<B: Backend>(&self, mut sample: Sample<B>) -> Result<Sample<B>, Error> {
        sample.image = image_ops::contrast(sample.image, self.value);
        Ok(sample)
    }
}

pub struct Brighten {
    value: f32,
}

impl Transform for Brighten {
    fn apply_image<B: Backend>(&self, mut sample: Sample<B>) -> Result<Sample<B>, Error> {
        sample.image = image_ops::hue_rotate(sample.image, self.value);
        Ok(sample)
    }
}

/// doc
pub struct HueRotate {
    angle: f32,
}

impl Transform for HueRotate {
    fn apply_image<B: Backend>(&self, mut sample: Sample<B>) -> Result<Sample<B>, Error> {
        sample.image = image_ops::hue_rotate(sample.image, self.angle);
        Ok(sample)
    }
}

#[cfg(test)]
mod tests {
    use burn_ndarray::NdArray;
    use std::hash::{DefaultHasher, Hash, Hasher};

    use super::*;

    fn create_sample_from_image_dataset_item() -> Sample<NdArray> {
        let img = image_ops::create_test_image(12, 12, [127, 128, 255]);
        let img = img.iter().map(|&x| PixelDepth::U8(x)).collect();

        let bb_1 = BoundingBox {
            coords: [1.0, 1.0, 6.0, 6.0],
            label: 0,
        };
        let bb_2 = BoundingBox {
            coords: [1.0, 2.0, 3.0, 4.0],
            label: 1,
        };

        let idi = ImageDatasetItem {
            image: img,
            image_dims: (32, 32),
            annotation: Annotation::BoundingBoxes(vec![bb_1, bb_2]),
            image_path: "/nowhere/".into(),
        };

        idi.into_sample()
    }

    fn create_sample_from_test_img() -> Sample<NdArray> {
        let img = image_ops::create_test_image(12, 12, [127, 128, 255]);

        let bb_1 = BoundingBox {
            coords: [1.0, 1.0, 6.0, 6.0],
            label: 0,
        };
        let bb_2 = BoundingBox {
            coords: [1.0, 2.0, 3.0, 4.0],
            label: 1,
        };

        let ann = Annotation::BoundingBoxes(vec![bb_1, bb_2]);

        let img = img.iter().map(|&x| PixelDepth::U8(x)).collect();

        let item = ImageDatasetItem {
            image: img,
            image_dims: (12, 12),
            annotation: ann,
            image_path: "./nowhere".into(),
        };

        item.into_sample()
    }

    #[test]
    fn as_sample_test() {
        let sample = create_sample_from_test_img();
        dbg!(sample);
    }

    #[test]
    fn vertical_flip_test() {
        let sample = create_sample_from_image_dataset_item();

        let vf = VerticalFlip {};

        let sample = vf.apply(sample).unwrap();

        let AnnotationTnsr::BoundingBoxes(bb_vec) = &sample.annotations[0] else {
            panic!("Message type wrong")
        };

        // // Test hash of image
        let test_success_hash: u64 = 10732386221966926898;
        let mut h = DefaultHasher::new();
        sample.image.to_data().as_bytes().hash(&mut h);
        assert_eq!(test_success_hash, h.finish());

        // Check bounding box translations

        let eq_test_t = bb_vec[0].to_data();
        let eq_test_t: Vec<i32> = eq_test_t
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|&x| x as i32)
            .collect();

        assert_eq!(eq_test_t.as_slice(), [1, 5, 6, 6]);

        let eq_test_t = bb_vec[1].to_data();
        let eq_test_t: Vec<i32> = eq_test_t
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|&x| x as i32)
            .collect();

        assert_eq!(eq_test_t.as_slice(), [1, 6, 3, 4]);
    }

    #[test]
    fn imagedatasetitem_as_sample_test() {
        let idsi_sample = create_sample_from_image_dataset_item();
        dbg!(idsi_sample);
    }

    #[test]
    fn zoom_test() {
        let sample = create_sample_from_image_dataset_item();
        let zm = Zoom {
            fill: 8,
            top: 4,
            bottom: 6,
            right: 8,
            left: 9,
        };

        let sample = zm.apply(sample);
    }

    #[test]
    fn composition_test() {
        let sample = create_sample_from_image_dataset_item();
        let zm =  Zoom {
            fill: 8,
            top: 4,
            bottom: 6,
            right: 8,
            left: 9,
        };

    }
}
