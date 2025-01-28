//! Block-based komura equivalence, adapted from
//! S. Allegretti, F. Bolelli, C. Grana,
//! "Optimized Block-Based Algorithms to Label Connected Components on GPUs,"
//! in IEEE Transactions on Parallel and Distributed Systems, 2019.

use crate::{
    kernel,
    ops::numeric::{empty_device, zeros_device},
    tensor::JitTensor,
    tests::burn_tensor::{DType, Shape},
    JitElement, JitRuntime,
};
use cubecl::cube;
use cubecl::prelude::*;

mod info {
    pub const A: u8 = 0;
    pub const B: u8 = 1;
    pub const C: u8 = 2;
    pub const D: u8 = 3;
    pub const Q: u8 = 5;
    pub const R: u8 = 6;
    pub const S: u8 = 7;
}

#[cube]
fn has_bit<I: Int>(bitmap: I, pos: u8) -> bool {
    bool::cast_from((bitmap >> I::cast_from(pos)) & I::new(1))
}

#[cube]
fn set_bit<I: Int>(bitmap: I, pos: u8) -> I {
    bitmap | (I::new(1) << I::cast_from(pos))
}

#[cube]
fn find_root(s_buf: &Tensor<Atomic<u32>>, n: u32) -> u32 {
    let mut n = n;
    while Atomic::load(&s_buf[n]) != n {
        n = Atomic::load(&s_buf[n]);
    }
    n
}

#[cube]
fn find_root_and_compress(s_buf: &mut Tensor<u32>, id: u32) -> u32 {
    let mut n = id;
    while s_buf[n] != n {
        n = s_buf[n];
        s_buf[id] = n;
    }
    n
}

#[cube]
fn tree_union(s_buf: &Tensor<Atomic<u32>>, a: u32, b: u32) {
    let mut a = a;
    let mut b = b;
    #[allow(unused_assignments)]
    let mut done = false;

    loop {
        a = find_root(s_buf, a);
        b = find_root(s_buf, b);

        #[allow(clippy::comparison_chain, reason = "not supported in cubecl")]
        if a < b {
            let old = Atomic::min(&s_buf[b], a);
            done = old == b;
            b = old;
        } else if b < a {
            let old = Atomic::min(&s_buf[a], b);
            done = old == a;
            a = old;
        } else {
            done = true;
        }

        if done {
            break;
        }
    }
}

#[cube(launch)]
fn init_labeling(img: &Tensor<u8>, labels: &mut Tensor<i32>, last_pixel: &mut Array<u8>) {
    let batch = ABSOLUTE_POS_Z;
    let row = ABSOLUTE_POS_Y * 2;
    let col = ABSOLUTE_POS_X * 2;

    if row >= labels.shape(1) || col >= labels.shape(2) {
        terminate!();
    }

    let img_rows = img.shape(2);
    let img_cols = img.shape(3);
    let img_stride = img.stride(2);
    let labels_stride = labels.stride(1);

    let img_index = batch * img.stride(0) + row * img_stride + col * img.stride(3);
    let labels_index = batch * labels.stride(0) + row * labels_stride + col * labels.stride(2);

    let mut p = 0u16;

    // Bitmask representing two kinds of information
    // Bits 0, 1, 2, 3 are set if pixel a, b, c, d are foreground, respectively
    // Bits 4, 5, 6, 7 are set if block P, Q, R, S need to be merged to X in Merge phase
    let mut info = 0u8;

    let mut buffer = Array::<u8>::new(4);
    #[unroll]
    for i in 0..4 {
        buffer[i] = 0;
    }

    if col + 1 < img_cols {
        buffer[0] = img[img_index];
        buffer[1] = img[img_index + 1];

        if row + 1 < img_rows {
            buffer[2] = img[img_index + img_stride];
            buffer[3] = img[img_index + img_stride + 1];
        }
    } else {
        buffer[0] = img[img_index];

        if row + 1 < img_rows {
            buffer[2] = img[img_index + img_stride];
        }
    }

    if buffer[0] != 0 {
        p |= 0x777;
        info = set_bit::<u8>(info, info::A);
    }
    if buffer[1] != 0 {
        p |= 0x777 << 1;
        info = set_bit::<u8>(info, info::B);
    }
    if buffer[2] != 0 {
        p |= 0x777 << 4;
        info = set_bit::<u8>(info, info::C);
    }
    if buffer[3] != 0 {
        info = set_bit::<u8>(info, info::D);
    }

    if col == 0 {
        p &= 0xeeee;
    }
    if col + 1 >= img_cols {
        p &= 0x3333;
    } else if col + 2 >= img_cols {
        p &= 0x7777;
    }

    if row == 0 {
        p &= 0xfff0;
    }
    if row + 1 >= img_rows {
        p &= 0x00ff;
    } else if row + 2 >= img_rows {
        p &= 0x0fff;
    }

    // P is now ready to be used to find neighbor blocks
    // P value avoids range errors

    let mut father_offset = 0i32;

    // P square
    if has_bit::<u16>(p, 0) && img[img_index - img_stride - 1] != 0 {
        father_offset = -(2 * labels_stride as i32 + 2);
    }

    // Q square
    if (has_bit::<u16>(p, 1) && img[img_index - img_stride] != 0)
        || (has_bit::<u16>(p, 2) && img[img_index + 1 - img_stride] != 0)
    {
        if father_offset == 0 {
            father_offset = -(2 * labels_stride as i32);
        } else {
            info = set_bit::<u8>(info, info::Q);
        }
    }

    // R square
    if has_bit::<u16>(p, 3) && img[img_index + 2 - img_stride] != 0 {
        if father_offset == 0 {
            father_offset = -(2 * labels_stride as i32 - 2);
        } else {
            info = set_bit::<u8>(info, info::R);
        }
    }

    // S square
    if (has_bit::<u16>(p, 4) && img[img_index - 1] != 0)
        || (has_bit::<u16>(p, 8) && img[img_index + img_stride - 1] != 0)
    {
        if father_offset == 0 {
            father_offset = -2i32;
        } else {
            info = set_bit::<u8>(info, info::S);
        }
    }

    labels[labels_index] = labels_index as i32 + father_offset;
    if col + 1 < labels.shape(2) {
        labels[labels_index + 1] = info as i32;
    } else if row + 1 < labels.shape(1) {
        labels[labels_index + labels_stride] = info as i32;
    } else {
        last_pixel[0] = info;
    }
}

#[cube(launch)]
fn merge(labels: &mut Tensor<Atomic<u32>>, last_pixel: &mut Array<u8>) {
    let batch = ABSOLUTE_POS_Z;
    let row = ABSOLUTE_POS_Y * 2;
    let col = ABSOLUTE_POS_X * 2;
    let rows = labels.shape(1);
    let cols = labels.shape(2);
    let labels_stride = labels.stride(1);
    let labels_index = batch * labels.stride(0) + row * labels_stride + col;

    if row >= labels.shape(1) || col >= labels.shape(2) {
        terminate!();
    }

    let info = if col + 1 < cols {
        Atomic::load(&labels[labels_index + 1]) as u8
    } else if row + 1 < rows {
        Atomic::load(&labels[labels_index + labels_stride]) as u8
    } else {
        last_pixel[0]
    };

    if has_bit::<u8>(info, info::Q) {
        tree_union(labels, labels_index, labels_index - 2 * labels_stride);
    }
    if has_bit::<u8>(info, info::R) {
        tree_union(labels, labels_index, labels_index - 2 * labels_stride + 2);
    }
    if has_bit::<u8>(info, info::S) {
        tree_union(labels, labels_index, labels_index - 1);
    }
}

#[cube(launch)]
fn compression(labels: &mut Tensor<u32>) {
    let batch = ABSOLUTE_POS_Z;
    let row = ABSOLUTE_POS_Y * 2;
    let col = ABSOLUTE_POS_X * 2;
    let labels_index = batch * labels.stride(0) + row * labels.stride(1) + col;

    if row < labels.shape(1) && col < labels.shape(2) {
        find_root_and_compress(labels, labels_index);
    }
}

#[cube(launch)]
fn final_labeling(img: &Tensor<u8>, labels: &mut Tensor<u32>) {
    let batch = ABSOLUTE_POS_Z;
    let row = ABSOLUTE_POS_Y * 2;
    let col = ABSOLUTE_POS_X * 2;
    let rows = labels.shape(1);
    let cols = labels.shape(2);
    let label_stride = labels.stride(1);
    let img_stride = img.stride(2);
    let labels_index = batch * labels.stride(0) + row * label_stride + col;

    if row >= labels.shape(1) || col >= labels.shape(2) {
        terminate!();
    }

    let mut label = 0;
    #[allow(unused_assignments)]
    let mut info = 0u8;
    let mut buffer = Array::<u32>::new(2);

    if col + 1 < cols {
        buffer[0] = label[labels_index];
        buffer[1] = label[labels_index + 1];
        label = buffer[0] + 1;
        info = buffer[1] as u8;
    } else {
        label = labels[labels_index] + 1;
        if row + 1 < rows {
            info = labels[labels_index + label_stride] as u8;
        } else {
            // Read from the input image
            // "a" is already in position 0
            info = img[batch * img.stride(0) + row * img_stride + col];
        }
    }

    if col + 1 < cols {
        labels[labels_index] = has_bit::<u8>(info, info::B) as u32 * label;
        labels[labels_index + 1] = has_bit::<u8>(info, info::A) as u32 * label;

        if row + 1 < rows {
            labels[labels_index + label_stride] = has_bit::<u8>(info, info::D) as u32 * label;
            labels[labels_index + label_stride + 1] = has_bit::<u8>(info, info::C) as u32 * label;
        }
    } else {
        labels[labels_index] = has_bit::<u8>(info, info::A) as u32 * label;

        if row + 1 < rows {
            labels[labels_index + label_stride] = has_bit::<u8>(info, info::C) as u32 * label;
        }
    }
}

#[expect(
    unused,
    reason = "currently broken because kernel reassigns pointers and I need to figure out how to port that"
)]
pub fn block_based_komura_equivalence<R: JitRuntime, EI: JitElement>(
    img: JitTensor<R>,
) -> JitTensor<R> {
    let img = kernel::cast::<R, EI, u8>(img);

    let [batches, channels, rows, columns] = img.shape.dims();
    assert_eq!(channels, 1, "Channels must be 1 for connected components");

    let shape = Shape::new([batches, rows, columns]);
    let labels = zeros_device::<R, u32>(img.client.clone(), img.device.clone(), shape);

    let last_pixel = if (rows == 1 || columns == 1) && (rows + columns) % 2 == 0 {
        empty_device::<R, u8>(img.client.clone(), img.device.clone(), Shape::new([1]))
    } else {
        let offset = (((rows - 2) * labels.strides[2]) + (columns - 2)) * size_of::<u32>();
        JitTensor::new_contiguous(
            labels.client.clone(),
            labels.device.clone(),
            Shape::new([1]),
            labels.handle.clone().offset_start(offset as u64),
            DType::U8,
        )
    };

    let cube_dim = CubeDim::default();
    let cube_count_x = (columns as u32).div_ceil(2).div_ceil(cube_dim.x);
    let cube_count_y = (rows as u32).div_ceil(2).div_ceil(cube_dim.y);
    let cube_count = CubeCount::Static(cube_count_x, cube_count_y, batches as u32);

    init_labeling::launch(
        &img.client,
        cube_count.clone(),
        cube_dim,
        img.as_tensor_arg::<u8>(1),
        labels.as_tensor_arg::<u32>(1),
        last_pixel.as_array_arg::<u8>(1),
    );

    compression::launch(
        &img.client,
        cube_count.clone(),
        cube_dim,
        labels.as_tensor_arg::<u32>(1),
    );

    merge::launch(
        &img.client,
        cube_count.clone(),
        cube_dim,
        labels.as_tensor_arg::<u32>(1),
        last_pixel.as_array_arg::<u8>(1),
    );

    compression::launch(
        &img.client,
        cube_count.clone(),
        cube_dim,
        labels.as_tensor_arg::<u32>(1),
    );

    final_labeling::launch(
        &img.client,
        cube_count.clone(),
        cube_dim,
        img.as_tensor_arg::<u8>(1),
        labels.as_tensor_arg::<u32>(1),
    );

    labels
}
