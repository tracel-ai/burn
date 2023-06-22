@group(0)
@binding(0)
var<storage, read> lhs: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read> rhs: array<{{ elem }}>;

@group(0)
@binding(2)
var<storage, read_write> output: array<{{ elem }}>;

@group(0)
@binding(3)
var<storage, read> info: array<u32>;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_x }}, {{ workgroup_size_z }}) // ligne 241. 16, 16, 1
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Indexes
    //  [20, 10, 5, 256, 256] * [20, 10, 5, 256, 256]
    // [..., 256, 256] -> 16 instanciations de 16 threads
    let row = global_id.x; // va jusqu'à 255
    let col = global_id.y; // va jusqu'à 255
    let batch = global_id.z;  // va jusqu'à 999
    // [2, 2] -> 0, 1, 2, 3
    // [20, 10, 5] -> 0, 1, 2, ..., 999

    // Basic information
    let dim = info[0];
    let n_rows = info[6u * dim - 1u];
    let n_cols = info[6u * dim];
    let K = info[5u * dim - 1u];

    // Returns if outside the output dimension
    if row >= n_rows || col >= n_cols {
        return;
    }

    // Calculate the corresponding offsets with support for broadcasting.
    let offset_output = batch * n_rows * n_cols;
    var offset_lhs: u32 = 0u;
    var offset_rhs: u32 = 0u;

    let batch_dims = dim - 2u;
    for (var b: u32 = 1u; b <= batch_dims; b++) {
        let stride_lhs = info[b];
        let stride_rhs = info[b + 1u * dim];
        let stride_output = info[b + 2u * dim];
        let shape_lhs = info[b + 3u * dim];
        let shape_rhs = info[b + 4u * dim];

        offset_lhs += offset_output / stride_output % shape_lhs * stride_lhs;
        offset_rhs += offset_output / stride_output % shape_rhs * stride_rhs;
    }

    // Basic matmul implementation
    var sum = 0.0;
    for (var k: u32 = 0u; k < K; k++) {
        let lhs_index = row * K + k;
        let rhs_index = k * n_cols + col;

        sum += lhs[offset_lhs + lhs_index] * rhs[offset_rhs + rhs_index];
    }

    let output_index = row * n_cols + col;
    output[offset_output + output_index] = sum;
}
