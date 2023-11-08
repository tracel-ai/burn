@group(0)
@binding(0)
var<storage, read> input: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read_write> output: array<{{ elem }}>;

@group(0)
@binding(2)
var<storage, read> info: array<u32>;

var<workgroup> shared_memory: array<{{ elem }}, {{ shared_size }}>;

const WORKGROUP_SIZE_X = {{ workgroup_size_x }}u;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(local_invocation_id) local_invocation_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let workgroup_size_x = {{ workgroup_size_x }}u;
    let workgroup_size_y = {{ workgroup_size_y }}u;
    
    // Which sum_group (not position, but absolute id)
    let sum_group_id = workgroup_id.y * num_workgroups.x + workgroup_id.x;
    // nth thread in the sum_group
    let local_id = local_invocation_id.y * workgroup_size_x + local_invocation_id.x;
    var n_threads = workgroup_size_x * workgroup_size_y;

    // rank of the tensors
    let rank: u32 = info[0];
    // dimension on which to reduce (in 0..rank)
    let dim_reduce = info[4u * rank + 1u];
    // threads are responsible of how many inputs at first
    let n_input_values_per_thread = info[4u * rank + 2u];
    let reduce_factor = info[4u * rank + 3u];

    let stride_input_dim_reduce = info[dim_reduce + 1u];
    let shape_input_dim_reduce = info[dim_reduce + 1u + 2u * rank];

    var index_offset: u32 = 0u;

    // we must use sum_group_id to deduce coordinates (with hole at dim_reduce)
    // of the sum_group
    for (var i: u32 = 0u; i < rank; i++) {
        let stride_input = info[i + 1u];
        let stride_output = info[i + 1u + rank];
        let shape_input = info[i + 1u + 2u * rank]; // useless
        let shape_output = info[i + 1u + 3u * rank];

        let num_block = sum_group_id / stride_output % shape_output;
        index_offset += num_block * stride_input;
        
        // normally shape output should be 1 for dim_reduce, leading to num_block=0 and no offset
        // so we do not need to make a special case
    }

    // index_offset is the absolute position of the first element of the sum_group
    // the other elements should be i * stride_input_dim_reduce farther
    // this thread should work with positions
    // let first_input = index_offset + local_id * stride_input_dim_reduce;
    // actually first_input + i * n_threads * stride_input_dim_reduce
    // with i from 0 to n_input_values_per_thread (exclude)

    // then a for loop on n_input_values_per_thread, 
    // skipping n_threads between each time to maybe get some coalescing
    
    // ensure we start with zeros
    shared_memory[local_id] = 0.0;

    for (var i = 0u; i < n_input_values_per_thread; i++) {
        let nth = local_id + i * n_threads;
        if nth < shape_input_dim_reduce {
            let current_position = index_offset + nth * stride_input_dim_reduce;
            let value = input[current_position];
            shared_memory[local_id] += value; // += ok because should be the only thread working on this index
            // but beware, it will be different for other reductions
        }
    }

    workgroupBarrier();

    // probably stupid that every thread goes inside the loop
    while n_threads > 1u {
        n_threads /= reduce_factor;

        if local_id < n_threads {
            let write_position = local_id;
            for (var i = 1u; i < reduce_factor; i++) {
                let read_position = write_position + i * n_threads;
                let value = shared_memory[read_position];
                shared_memory[write_position] += value;
            }
        } // else it's idle

        workgroupBarrier();
    }
    
    if local_id == 0u {
        let output_position = sum_group_id; // not sure 
        output[output_position] = shared_memory[0u];
    }
}
