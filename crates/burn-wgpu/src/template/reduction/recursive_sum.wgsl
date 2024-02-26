const WORKGROUP_SIZE = {{ workgroup_size }}u;
const WORKGROUP_SIZE_X = {{ workgroup_size_x }}u;
const WORKGROUP_SIZE_Y = {{ workgroup_size_y }}u;

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
    let last_index = (WORKGROUP_SIZE_Y * num_workgroups.y) * (WORKGROUP_SIZE_X * num_workgroups.x) - 1u;

    data[id_local] = input[id_global];

    workgroupBarrier();

    var num_data = WORKGROUP_SIZE;
    if (id_local == last_index) {
        num_data = min(WORKGROUP_SIZE, arrayLength(&input) % WORKGROUP_SIZE);
        if (num_data == 0u) {
            num_data = WORKGROUP_SIZE;
        }
    }

    if id_local == 0u {
        var sum = {{ elem }}(0);
        for (var i: u32 = 0u; i < num_data; i++) {
            sum += data[i];
        }

        let id_output = workgroup_id.y * num_workgroups.x + workgroup_id.x;
        output[id_output] = sum;
    }
}
