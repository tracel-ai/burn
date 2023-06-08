@group(0)
@binding(0)
var<storage, read> lhs: array<elem>;

@group(0)
@binding(1)
var<storage, read> rhs: array<elem>;

@group(0)
@binding(2)
var<storage, read_write> output: array<elem>;

@group(0)
@binding(3)
var<storage, read> info: array<u32>;

var<workgroup> mds: array<array<elem, TILE_SIZE>, TILE_SIZE>;
var<workgroup> nds: array<array<elem, TILE_SIZE>, TILE_SIZE>;

@compute
@workgroup_size(WORKGROUP_SIZE_X, TILE_SIZE, TILE_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    // Indexes
    let batch = global_id.x;
    let row = global_id.y;
    let col = global_id.z;

    // Basic information
    let dim = info[0];
    let n_rows = info[3u * dim - 1u];
    let n_cols = info[4u * dim];
    let K = info[3u * dim];
    let is_outside = row >= n_rows || col >= n_cols;

    // Calculate the corresponding offsets with support for broadcasting.
    let offset_output = batch * n_rows * n_cols;
    var offset_lhs: u32 = 0u;
    var offset_rhs: u32 = 0u;

    let batch_dims = dim - 2u;
    for (var b: u32 = 0u; b < batch_dims; b++) {
        let stride_lhs = info[b + 1u];
        let stride_rhs = info[b + 1u * dim + 1u];
        let shape_lhs = info[b + 2u * dim + 1u];
        let shape_rhs = info[b + 3u * dim + 1u];

        offset_lhs += offset_output / stride_lhs % shape_lhs * stride_lhs;
        offset_rhs += offset_output / stride_rhs % shape_rhs * stride_rhs;
    }

    var sum = elem(0);
    let num_tiles = u32(ceil(f32(K) / f32(TILE_SIZE)));

    for (var m = 0u; m < num_tiles; m++) {
        let lhs_index = row * K + (m * TILE_SIZEu + local_id.z);
        let rhs_index = (m * TILE_SIZEu + local_id.y) * K + col;

        mds[local_id.y][local_id.z] = lhs[lhs_index];
        nds[local_id.y][local_id.z] = rhs[rhs_index];

        workgroupBarrier();

        for (var k = 0u; k < TILE_SIZEu; k++) {
            sum += mds[local_id.y][k] * nds[k][local_id.z];
        }

        workgroupBarrier();
    }

    // Returns if outside the output dimension
    // Important to be after all workgroupBarrier()
    if is_outside {
        return;
    }

    let output_index = row * n_rows + col;
    output[offset_output + output_index] = sum;
}
