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


const BLOCK_SIZE = {{ workgroup_size_x }}u; //

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>, // 0.. nombre d'invocations * nombre de threads par invocation
    @builtin(local_invocation_index) local_idx: u32, // 0..nombre de threads par invocation -> threadIdx dans Cuda
    @builtin(workgroup_id) workgroup_id: vec3<u32>, // 0.. nombre d'invocations -> blockIdx dans Cuda
) {
// @compute
// @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_x }}, {{ workgroup_size_z }})
// fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Indexes
    //  [20, 10, 5, 200, 50] * [20, 10, 5, 50, 100]
    // [..., 256, 256] -> 16 instanciations de 16 threads
    // let row = global_id.x; // va jusqu'à 199 -> suppose 103
    // let col = global_id.y; // va jusqu'à 99  -> suppose 23
    // let batch = global_id.z;  // va jusqu'à 999 -> suppose 436
    // [2, 2] -> 0, 1, 2, 3
    // [20, 10, 5] -> 0, 1, 2, ..., 999

    // Indexes
    // Le bloc est un carré BLOCK_SIZE x BLOCK_SIZE
    // workgroup_id.x * BLOCK_SIZE -> skipper les blocs qui ne nous concernent pas (dimension row)
    // local_idx / BLOCK_SIZE -> quelle rangée dans le bloc
    let row = workgroup_id.x * BLOCK_SIZE + (local_idx / BLOCK_SIZE); 
    // workgroup_id.y * BLOCK_SIZE -> skipper les blocs qui ne nous concernent pas (dimension col)
    // local_idx % BLOCK_SIZE -> quelle colonne dans le bloc
    let col = workgroup_id.y * BLOCK_SIZE + (local_idx % BLOCK_SIZE);
    let batch = global_id.z;


    // Indexes
    // let row = global_id.x;  // per thread, which output place its working on
    // let col = global_id.y;  // for each dimension
    // let batch = global_id.z; //which batch it's working on, merged

    // Basic information
    let dim = info[0];   // equal to D, rank. 5
    let n_rows = info[6u * dim - 1u];  // output tensor number of rows 200
    let n_cols = info[6u * dim]; // output tensor number of columns 100
    let K = info[5u * dim - 1u]; // common dimension between the two outputs 50

    // Returns if outside the output dimension. Can happen because we use ceiling in rust matmul
    if row >= n_rows || col >= n_cols {
        return;
    }

    // Calculate the corresponding offsets with support for broadcasting.
    let offset_output = batch * n_rows * n_cols; // skip all the preceding batches. -> 436 * 100 * 200 = 8720000
    // if we see it as a long vector it's obvious

    var offset_lhs: u32 = 0u;
    var offset_rhs: u32 = 0u;

    let batch_dims = dim - 2u; // how many batch ranks. 3
    for (var b: u32 = 1u; b <= batch_dims; b++) { // for each of those batch ranks
        let stride_lhs = info[b]; // the stride for that batch dim on lhs [500000, 50000, 10000]
        let stride_rhs = info[b + 1u * dim]; // the stride for that batch dim on rhs [250000, 25000, 5000]
        let stride_output = info[b + 2u * dim]; // the stride on that batch dim on output [1000000, 100000, 20000]
        let shape_lhs = info[b + 3u * dim]; // the length of that batch dim on lhs [20, 10, 5]
        let shape_rhs = info[b + 4u * dim]; // the length of that batch dim on rhs [20, 10, 5]
        // what's the length of that batch dim on output? normally := shape_lhs = shape_rhs [20, 10, 5]

        offset_lhs += offset_output / stride_output % shape_lhs * stride_lhs;
        // 8720000 / 1000000 -> 8
        // 8 % 20 -> 8
        // 8 * 500000 -> 4000000
        // 8720000 / 100000 -> 87
        // 87 % 10 -> 7
        // 7 * 50000 -> 350000
        // 8720000 / 20000 -> 436
        // 436 % 5 -> 1
        // 1 * 50000 -> 10000
        // off_set = 4000000+350000+10000 = 4360000, un multiple de 20000 qui est le nombre d'éléments par batch
        offset_rhs += offset_output / stride_output % shape_rhs * stride_rhs;
        // 8720000 / 1000000 -> 8
        // 8 % 20 -> 8
        // 8 * 250000 -> 2000000
        // 8720000 / 100000 -> 87
        // 87 % 10 -> 7
        // 7 * 25000 -> 175000
        /// 8720000 / 20000 -> 436
        // 436 % 5 -> 1
        // 1 * 5000 -> 5000
        // off_set = 2000000+175000+5000=2180000, un multiple de 20000 qui est le nombre d'éléments par batch
    }

    // Basic matmul implementation
    var sum = 0.0;
    for (var k: u32 = 0u; k < K; k++) {
        let lhs_index = row * K + k; //      = row * stride_lhs[-2] +   k * stride_lhs[-1]
        let rhs_index = k * n_cols + col; // = k   * stride_rhs[-2] + col * stride_lhs[-1]

        sum += lhs[offset_lhs + lhs_index] * rhs[offset_rhs + rhs_index];
    }


    let output_index = row * n_cols + col; // = row * stride_output[-2] + col * stride_output[-1]
    // 

    output[offset_output + output_index] = sum;
}
