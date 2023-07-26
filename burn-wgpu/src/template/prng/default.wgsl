@group(0)
@binding(0)
var<storage, read_write> output: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read> info: array<u32>;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let id = global_id.y * (num_workgroups.x * {{ workgroup_size_x }}u) + global_id.x;
    let global_seed: u32 = info[0];
    let thread_seed = global_seed / 1000u + id; // weird behaviour

    output[id] = f32(thread_seed);
}

