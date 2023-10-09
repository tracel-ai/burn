@group(0)
@binding(0)
var<storage, read_write> weight: array<{{ elem }}>; // weight matrix

@group(0)
@binding(1)
var<storage, read> info: array<u32, 32>;

const WORKGROUP_SIZE_X = {{ workgroup_size_x }}u;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let linear_id = global_id.y * (num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;
    let in_channels = info[5]; //  or is this 6?
    let kernel_size_0 = info[9]; 
    let kernel_size_1 = info[10];  

    var remainder = linear_id;

    let out_c = remainder / (in_channels * kernel_size_0 * kernel_size_1);
    remainder = remainder % (in_channels * kernel_size_0 * kernel_size_1);

    let c = remainder / (kernel_size_0 * kernel_size_1);
    remainder = remainder % (kernel_size_0 * kernel_size_1);

    let kh = remainder / kernel_size_1;
    let kw = remainder % kernel_size_1;

    let output_idx = out_c * in_channels * kernel_size_0 * kernel_size_1 + c * kernel_size_0 * kernel_size_1 + kh * kernel_size_1 + kw;

    if (c < in_channels && kh < kernel_size_0 && kw < kernel_size_1) {
        weight[output_idx] = 1.0
    }
}
