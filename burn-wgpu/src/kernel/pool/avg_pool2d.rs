use crate::{
    element::WgpuElement,
    kernel::{
        self, elemwise_workgroup,
        pool::{build_output_and_info_pool2d, build_pool2d_info},
        KernelSettings,
    },
    kernel_wgsl,
    tensor::WgpuTensor,
};

kernel_wgsl!(AvgPool2d, "../../template/pool/avg_pool2d.wgsl");
kernel_wgsl!(
    AvgPool2dBackward,
    "../../template/pool/avg_pool2d_backward.wgsl"
);

pub(crate) fn avg_pool2d<E: WgpuElement>(
    x: WgpuTensor<E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> WgpuTensor<E, 4> {
    const WORKGROUP: usize = 32;

    let (info_buffer, output) = build_output_and_info_pool2d(&x, kernel_size, stride, padding);
    let kernel = x
        .context
        .compile_static::<KernelSettings<AvgPool2d, E, i32, WORKGROUP, WORKGROUP, 1>>();

    x.context.execute(
        elemwise_workgroup(output.shape.num_elements(), WORKGROUP),
        kernel,
        &[&x.buffer, &output.buffer, &info_buffer],
    );

    output
}

pub(crate) fn avg_pool2d_backward<E: WgpuElement>(
    x: WgpuTensor<E, 4>,
    grad: WgpuTensor<E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> WgpuTensor<E, 4> {
    const WORKGROUP: usize = 32;

    let grad = kernel::into_continuous(grad);

    let num_elems = x.shape.num_elements();
    let buffer = x
        .context
        .create_buffer(num_elems * core::mem::size_of::<E>());
    let output = WgpuTensor::new(x.context.clone(), x.shape.clone(), buffer);
    let info_buffer = build_pool2d_info(&x, &grad, kernel_size, stride, padding);
    let kernel = x
        .context
        .compile_static::<KernelSettings<AvgPool2dBackward, E, i32, WORKGROUP, WORKGROUP, 1>>();

    let grad_d = x.context.read_buffer(grad.buffer.clone());
    let grad_d: &[f32] = bytemuck::cast_slice(&grad_d);
    println!("{grad_d:?}");
    let info = x.context.read_buffer(info_buffer.clone());
    let info: &[u32] = bytemuck::cast_slice(&info);
    for xx in 0..WORKGROUP {
        for yy in 0..WORKGROUP {
            let id = (xx * WORKGROUP + yy) as u32;

            if id >= grad.shape.num_elements() as u32 {
                continue;
            }

            let input_stride_0 = info[0];
            let input_stride_1 = info[1];
            let input_stride_2 = info[2];
            let input_stride_3 = info[3];
            let input_shape_0 = info[4];
            let input_shape_1 = info[5];
            let input_shape_2 = info[6];
            let input_shape_3 = info[7];

            let grad_stride_0 = info[8];
            let grad_stride_1 = info[9];
            let grad_stride_2 = info[10];
            let grad_stride_3 = info[11];
            let grad_shape_0 = info[12];
            let grad_shape_1 = info[13];
            let grad_shape_2 = info[14];
            let grad_shape_3 = info[15];

            let kernel_size_0 = info[16];
            let kernel_size_1 = info[17];
            let pool_stride_0 = info[18];
            let pool_stride_1 = info[19];
            let padding_0 = info[20];
            let padding_1 = info[21];

            let b = id / input_stride_0 % input_shape_0;
            let c = id / input_stride_1 % input_shape_1;
            let ih = id / input_stride_2 % input_shape_2;
            let iw = id / input_stride_3 % input_shape_3;

            // The maximum number of overlapping filters that may content the current index.
            let kms_0 = kernel_size_0 as i32 - pool_stride_0 as i32;
            let kms_1 = kernel_size_1 as i32 - pool_stride_1 as i32;

            let oh_start_tmp = ((ih + padding_0) as i32 - kms_0 as i32) / pool_stride_0 as i32;
            let ow_start_tmp = ((iw + padding_1) as i32 - kms_1 as i32) / pool_stride_1 as i32;

            let oh_start = (i32::max(oh_start_tmp, 0)) as u32;
            let ow_start = (i32::max(ow_start_tmp, 0)) as u32;

            let oh_end = (i32::max(kms_0, 0)) as u32 + oh_start + 1;
            let ow_end = (i32::max(kms_1, 0)) as u32 + ow_start + 1;

            println!("===============================");
            println!("{id} | [{b},{c},{ih},{iw}]");
            let mut grad_acc = 0.0;

            let ih_min = oh_start * pool_stride_0;
            let iw_min = ow_start * pool_stride_1;
            let ih_max = ih_min + kernel_size_0;
            let iw_max = iw_min + kernel_size_1;
            // We iterate over each potentially resulting overlapping filters and check
            // if their max index is the current one.
            for oh in oh_start..oh_end {
                for ow in ow_start..ow_end {
                    println!("[{oh}, {ow}]");
                    println!("ih {ih_min}..{ih_max}");
                    println!("iw {iw_min}..{iw_max}");
                    if ih < ih_min || ih >= ih_max || oh >= grad_shape_2 {
                        println!("IH SKIP");
                        continue;
                    }
                    if iw < iw_min || iw >= iw_max || ow >= grad_shape_3 {
                        println!("IW SKIP");
                        continue;
                    }

                    let index = b * grad_stride_0
                        + c * grad_stride_1
                        + oh * grad_stride_2
                        + ow * grad_stride_3;
                    let val = grad_d[index as usize];
                    println!("{val}");
                    grad_acc += val;
                }
            }
        }
    }

    x.context.execute(
        elemwise_workgroup(output.shape.num_elements(), WORKGROUP),
        kernel,
        &[&grad.buffer, &output.buffer, &info_buffer],
    );

    output
}
