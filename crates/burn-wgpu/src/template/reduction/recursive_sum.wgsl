const WORKGROUP_SIZE = {{ workgroup_size }}u;
const WORKGROUP_SIZE_X = {{ workgroup_size_x }}u;

@group(0)
@binding(0)
var<storage, read> input: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read_write> output: array<{{ elem }}>;

var<workgroup> data: array<{{ elem }}, WORKGROUP_SIZE>;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let id_global = global_id.y * (num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;
    let id_local = local_id.y * WORKGROUP_SIZE_X + local_id.x;

    data[id_local] = input[id_global];

    workgroupBarrier();

    if id_local == 0u {
        var sum = {{ elem }}(0);
        for (var i: u32 = 0u; i < WORKGROUP_SIZE; i++) {
            sum += data[i];
        }

        let id_output = workgroup_id.y * num_workgroups.x + workgroup_id.x;
        output[id_output] = sum;
    }
}
