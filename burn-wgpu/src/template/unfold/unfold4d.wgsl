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

    let in_channels = info[5];
    let kernel_size_0 = info[9]; 
    let kernel_size_1 = info[10];  

    // Determine the current channel and position within kernel
    let c = linear_id / (kernel_size_0 * kernel_size_1);
    let kh = (linear_id % (kernel_size_0 * kernel_size_1)) / kernel_size_1;
    let kw = linear_id % kernel_size_1;

    // Initialize index for the output tensor
    var output_idx = c * (kernel_size_0 * kernel_size_1 * in_channels) + kh * (kernel_size_1 * in_channels) + kw * in_channels;

    // Set the appropriate locations to one
    if (c < in_channels && kh < kernel_size_0 && kw < kernel_size_1) {
        let output_channel = c * kernel_size_0 * kernel_size_1 + kh * kernel_size_1 + kw;
        weight[output_idx] = 1.0;
    }
}
