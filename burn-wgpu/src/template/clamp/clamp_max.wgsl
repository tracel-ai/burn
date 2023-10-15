@group(0)
@binding(0)
var<storage, read_write> input: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read> value: {{ elem }};

@group(0)
@binding(2)
var<storage, read> length: u32;

const WORKGROUP_SIZE_X = {{ workgroup_size_x }}u;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let linear_id = global_id.y * (num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;

    if(linear_id < length) {
        if(input[linear_id] > value) {
            input[linear_id] = value;
        }
    }
}