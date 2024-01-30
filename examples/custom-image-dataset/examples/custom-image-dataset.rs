use burn::data::dataset::{
    vision::{ImageFolderDataset, ImageTarget},
    Dataset,
};
use custom_image_dataset::dataset::CIFAR10Loader;

use image::{DynamicImage, RgbImage};

fn main() {
    let index = 0;
    let dataset = ImageFolderDataset::cifar10_test();

    println!("Dataset loaded with {} images", dataset.len());

    // Display first element label
    let item = dataset.get(index).unwrap();
    if let ImageTarget::Label(y) = item.target {
        println!("Element {} has label {}", index, y);
    }
}
