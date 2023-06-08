@group(0)
@binding(0)
var<storage, read> input: array<elem>;

@group(0)
@binding(1)
var<storage, read_write> output: array<elem>;

var<workgroup> data: array<elem, WORKGROUP_SIZE_X>;

@compute
@workgroup_size(WORKGROUP_SIZE_X, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    data[local_id.x] = input[global_id.x];

    workgroupBarrier();

    if local_id.x == 0u {
        var sum = elem(0);
        for (var i: u32 = 0u; i < WORKGROUP_SIZE_Xu; i++) {
            sum += data[i];
        }

        output[workgroup_id.x] = sum;
    }
}
