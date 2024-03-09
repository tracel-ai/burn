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
    
    // To determine which reduce_group (not position, but absolute id)
    let reduce_group_id = workgroup_id.y * num_workgroups.x + workgroup_id.x;

    // nth thread in the workgroup
    let local_id = local_invocation_id.y * workgroup_size_x + local_invocation_id.x;

    // rank of the tensors
    let rank: u32 = info[0];
    // dimension on which to reduce (in 0..rank)
    let dim_reduce = info[4u * rank + 1u];
    // threads are responsible of how many inputs in one reduce_group
    let n_input_values_per_thread = info[4u * rank + 2u];

    let stride_input_dim_reduce = info[dim_reduce + 1u];
    let shape_input_dim_reduce = info[dim_reduce + 1u + 2u * rank];
    var n_threads = workgroup_size_x * workgroup_size_y;

    var index_offset: u32 = 0u;
    
    for (var i: u32 = 0u; i < rank; i++) {
        let stride_input = info[i + 1u];
        let stride_output = info[i + 1u + rank];
        let shape_output = info[i + 1u + 3u * rank];

        let num_block = reduce_group_id / stride_output % shape_output;
        index_offset += num_block * stride_input;
    }

    // Ensure shared memory starts at 0
    shared_memory[local_id] = {{ elem }}(0);

    for (var i = 0u; i < n_input_values_per_thread; i++) {
        let nth = local_id + i * n_threads;
        if nth < shape_input_dim_reduce {
            let current_position = index_offset + nth * stride_input_dim_reduce;
            let value = input[current_position];
            
            {{ update }}
        }
    }

    workgroupBarrier();

    let reduce_factor = 2u; 
    while n_threads > 1u {
        n_threads /= reduce_factor;

        if local_id < n_threads {
            for (var i = 1u; i < reduce_factor; i++) {
                let read_position = local_id + i * n_threads;
                let value = shared_memory[read_position];
                
                {{ update }}
            }
        } 

        workgroupBarrier();
    }
    
    if local_id == 0u {
        let output_position = reduce_group_id; 
        let final_value = shared_memory[0u];

        {{ assign }}
    }
}
