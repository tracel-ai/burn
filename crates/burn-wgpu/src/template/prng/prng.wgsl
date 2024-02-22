@group(0)
@binding(0)
var<storage, read_write> output: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read> info: array<u32, 5>;

@group(0)
@binding(2)
var<storage, read> args: array<{{ elem }}, {{ num_args }}>;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(local_invocation_index) local_id: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    // Thread preparation
    let n_threads_per_workgroup = {{ workgroup_size }}u;
    let workgroup_offset = (workgroup_id.x * num_workgroups.y + workgroup_id.y) * n_threads_per_workgroup;
    let n_values_per_thread = info[0u];
    let write_index_base = workgroup_offset * n_values_per_thread + local_id; 

    // Set state with unique seeds
    let thread_seed = 1000000007u * (workgroup_offset + local_id);
    var state: array<u32, 4u>;
    for (var i = 0u; i < 4u; i++) {
        state[i] = info[i + 1u] + thread_seed;
    }

    // Creation of n_values_per_thread values, specific to the distribution 
    {{ prng_loop }}
}

fn taus_step(z: u32, s1: u32, s2: u32, s3: u32, m: u32) -> u32 {   
    let b = ((z << s1) ^ z) >> s2;  
    return ((z & m) << s3) ^ b; 
} 

fn taus_step_0(z: u32) -> u32 {
    return taus_step(z, 13u, 19u, 12u, 4294967294u);
}

fn taus_step_1(z: u32) -> u32 {
    return taus_step(z, 2u, 25u, 4u, 4294967288u);
}

fn taus_step_2(z: u32) -> u32 {
    return taus_step(z, 3u, 11u, 17u, 4294967280u);
}

fn lcg_step(z: u32) -> u32 {
    return (1664525u * z + 1013904223u);
}

fn cast_float(number: u32) -> {{ elem }} {
   return 2.3283064365387e-10 * {{ elem }}(number);
}
