# Data Gathering and Preprocessing
To train and use the unet example, data must be provided in the form of image pairs.

The training data used in the example can be downloaded from: [kaggle - nikhilroxtomar - brain-tumor-segmentation ](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation/data).

> [!NOTE]
> Not all images are the same resolution. Most are 512 x 512 pixels.

## Requirements
* Source files should be in an `images` directory, while corresponding target files are in a `masks` directory.
* All images have a corresponding mask with a consistent naming convention.
* All images and masks have a uniform size, that is a constant and consistent width and height.
* All masks have only two classes; class A is black (0,0,0) and class B is white (255,255,255).

Recommended directory structure:
|- unet
|  |- src
|  |- data
|     |- images
|        |- 1.png
|        |- 2.png
|        |- ...
|     |- masks
|        |- 1.png
|        |- 2.png
|        |- ...

## Dataset, Batcher, and DataLoader
The model takes a burn tensor with shape `[batch_size, channels, height, width]` as input and outputs a tensor with the shape `[batch_size, 1, height, width]`. The purpose of the data pipeline is to convert the images (png files) to tensors for training, testing, and validation.

Set the constants in the [`brain_tumor_data.rs`](./src/brain_tumor_data.rs) to match the data:

```rust
// height and width of the images used in training
pub const WIDTH: usize = 512;
pub const HEIGHT: usize = 512;
const TRAINING_DATA_DIRECTORY_STR: &str = "data";
```

To confirm the batching strategy, you can test the dataloader using:

```rust
use burn::data::dataloader::DataLoaderBuilder;
use unet::segmentation_data::BrainTumorBatcher;
use burn_unet_example::segmentation_data::BrainTumorDataset;

type MyBackend = NdArray<f32, i64, i8>;
let batcher_train = BrainTumorBatcher::<MyBackend>::new(device.clone());
let train_dataset: BrainTumorDataset =
    BrainTumorDataset::train().expect("Failed to build training dataset");
let dataloader_train = DataLoaderBuilder::new(batcher_train)
    .batch_size(2)
    .shuffle(42)
    .num_workers(1)
    .build(train_dataset);
println!("{}", dataloader_train.num_items());

for batch in dataloader_train.iter() {
    println!("training ...");
    println!("{:?}", batch.source_tensor.shape());
}
```

## Additional notes

#### Image to Tensor conversion
The following script illustrates how to convert an image file to a `burn::tensor::Tensor` with shape `[channels, height, width]` (aligns with Conv2D input).

```rust
// two ways to create tensors from images with shape [channel, height, width] (matching the expected Conv2d)
let p: std::path::PathBuf = std::path::Path::new("data_down4_n500_wh32")
    .join("images")
    .join("1_w32_h32.png");
let img: DynamicImage = image::open(p).expect("Failed to open image");
let width: usize = img.width() as usize;
let height: usize = img.height() as usize;
let mut v: Vec<f32> = Vec::<f32>::with_capacity(width * height * 3);
let rgb_img = match img {
    DynamicImage::ImageRgb8(rgb_img) => rgb_img.clone(),
    _ => img.to_rgb8(),
};
// Iterate over pixels and fill the array in parallel -> WRONG ORDER!!!!!!
//--- rgb_img
//---     .enumerate_pixels()
//---     .into_iter()
//---     .for_each(|(x, y, pixel)| {
//---         // Convert each channel from u8 (0-255) to f32 (0.0-1.0)
//---         v.push(pixel[0] as f32 / 255.0);
//---         v.push(pixel[1] as f32 / 255.0);
//---         v.push(pixel[2] as f32 / 255.0);
//---     });
// println!("{:?}", v);
for x in 0..32 {
    for y in 0..32 {
        let pixel = rgb_img.get_pixel(x, y);
        v.push(pixel[0] as f32 / 255.0);
        v.push(pixel[1] as f32 / 255.0);
        v.push(pixel[2] as f32 / 255.0);
    }
}
let a: Box<[f32]> = v.into_boxed_slice();
let d = TensorData::from(&*a);
let u: Tensor<MyBackend, 1> = Tensor::from_data(d, &device);
let u1 = u.reshape([width, height, 3]).swap_dims(0, 2);
println!("{:}", u1);
```

For small images, you can use an array, but keep in mind this is created on the stack and cause a stack overflow.

```rust
let mut r = [[[0.0f32; 32]; 32]; 3]; // large images will cause a stack-overflow here!
for x in 0..32 {
    for y in 0..32 {
        let pixel = rgb_img.get_pixel(x, y);
        let xx = x as usize;
        let yy = y as usize;
        // Convert each channel from u8 (0-255) to f32 (0.0-1.0)
        r[0][yy][xx] = pixel[0] as f32 / 255.0;
        r[1][yy][xx] = pixel[1] as f32 / 255.0;
        r[2][yy][xx] = pixel[2] as f32 / 255.0;
    }
}

let d = TensorData::from(r);
let u2: Tensor<MyBackend, 3> = Tensor::from_data(d, &device);
println!("{:}", u2);
```

To confirm both constructors are identical:

```rust
let close_enough = u1.all_close(u2, Some(1e-5), Some(1e-8));
println!("{:}", close_enough);
```

#### Downsampling
If memory is an issue, smaller input images will result in a smaller model. Downsample the images and masks with:

```rust
pub fn downsample_images(
    input_dir: &Path,
    output_dir: &Path,
    downscaling_factor: u32,
) -> Result<(), Error> {
    // Create output directory if it doesn't exist
    fs::create_dir_all(output_dir)?;

    let factor: u32 = 1u32 << downscaling_factor;

    // Iterate over the entries in the directory
    for entry in fs::read_dir(input_dir)? {
        let entry = entry?;
        let path = entry.path();

        // Check if the file has a .png extension
        if path.extension().and_then(|ext| ext.to_str()) == Some("png") {
            // Open the image file
            //let img: DynamicImage = ImageReader::open(&path)?.decode()?;
            let img: DynamicImage = ImageReader::open(&path)
                .map_err(|e| Error::new(ErrorKind::Other, e))?
                .decode()
                .map_err(|e| Error::new(ErrorKind::Other, e))?;

            // Downsample the image
            let downsampled_image = img.resize(
                img.width() / factor,
                img.height() / factor,
                FilterType::Nearest,
            );

            // Get new dimensions
            let (new_width, new_height) = downsampled_image.dimensions();

            // Create new filename with dimensions
            let original_stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unnamed");

            let new_filename = format!("{}_w{}_h{}.png", original_stem, new_width, new_height);

            // Create full output path
            let output_path = output_dir.join(new_filename);

            // Save the downsampled image
            downsampled_image
                .save(output_path)
                .map_err(|e| Error::new(ErrorKind::Other, e))?;
        }
    }

    Ok(())
}
```

For example,

```rust
let images_data_dir = Path::new("data").join("images");
let masks_data_dir = Path::new("data").join("masks");
let images_d4_data_dir = Path::new("data1").join("images");
let masks_d4_data_dir = Path::new("data1").join("masks");

let _ = utils::downsample_images(images_data_dir.as_ref(), images_d4_data_dir.as_ref(), 1);
let _ = utils::downsample_images(masks_data_dir.as_ref(), masks_d4_data_dir.as_ref(), 1);
```

_downscaling_factor_ mapping for input size 512 pixels:
* 0: 512/1 -> 512
* 1: 512/2 -> 256
* 2: 512/4 -> 128
* 3: 512/8 -> 64
* 4: 512/16 -> 32
* 5: 512/32 -> 16
