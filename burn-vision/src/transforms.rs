use burn_core::tensor::{backend::Backend, Data, Shape, Tensor};

pub fn pad_image<B: Backend>(tensor: Tensor<B, 3>, new_shape: [usize; 2]) -> Tensor<B, 3> {
    let old_shape = tensor.shape();
    let old_data = tensor.to_data().convert::<f32>().value;

    // Calculate padding - don't underflow
    let new_height = if new_shape[0] < old_shape.dims[1] {
        old_shape.dims[1]
    } else {
        new_shape[0]
    };
    let new_width = if new_shape[1] < old_shape.dims[2] {
        old_shape.dims[2]
    } else {
        new_shape[1]
    };

    println!("Old shape: {:?}", old_shape.dims);
    println!("New height: {}, New width: {}", new_height, new_width);

    let pad_height = (new_height - old_shape.dims[1]) / 2;
    let pad_width = (new_width - old_shape.dims[2]) / 2;

    let old_pixels = old_shape.dims[1] * old_shape.dims[2];
    let new_pixels = new_height * new_width;

    let mut new_data = vec![0.0; 3 * new_pixels];

    for i in 0..new_height {
        for j in 0..new_width {
            if pad_height <= i
                && i < pad_height + old_shape.dims[1]
                && pad_width <= j
                && j < pad_width + old_shape.dims[2]
            {
                let r = old_data[(i - pad_height) * old_shape.dims[2] + (j - pad_width)];
                let g =
                    old_data[(i - pad_height) * old_shape.dims[2] + (j - pad_width) + old_pixels];
                let b = old_data
                    [(i - pad_height) * old_shape.dims[2] + (j - pad_width) + 2 * old_pixels];

                new_data[i * new_width + j] = r;
                new_data[i * new_width + j + new_pixels] = g;
                new_data[i * new_width + j + 2 * new_pixels] = b;
            }
        }
    }
    let data = Data {
        value: new_data,
        shape: Shape::new([3, new_height, new_width]),
    };

    println!("New shape: {:?}", data.shape.dims);
    Tensor::<B, 3>::from_data(data.convert(), &tensor.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::ImageReader;
    use burn_core::backend::wgpu::{Wgpu, WgpuDevice};
    use std::env;
    use std::path::PathBuf;

    // Test reading and writing an image using the ImageReader.
    #[test]
    fn test_image_padding() {
        let device = WgpuDevice::default();
        let reader: ImageReader<Wgpu> = ImageReader::new(device.clone());

        let mut path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        path.push("src");
        path.push("tests");
        path.push("helicopter.png");
        println!("Reading image from: {:?}", path);

        let tensor = reader.read_image(path).unwrap();
        let tensor = pad_image(tensor, [1000, 700]);

        let mut path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        path.push("src");
        path.push("tests");
        path.push("helicopter_padding_test.png");
        reader.write_image(&tensor, path).unwrap();
    }
}
