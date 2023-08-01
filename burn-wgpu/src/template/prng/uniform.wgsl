@group(0)
@binding(0)
var<storage, read_write> output: array<{{ elem }}>;

@group(0)
@binding(1)
var<storage, read> info: array<u32, 5>;

@group(0)
@binding(2)
var<storage, read> args: array<{{ elem }}, 2>;

@compute
@workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, 1)
fn main(
    @builtin(local_invocation_index) local_id: u32,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let wg_size_x = {{ workgroup_size_x }}u;
    let wg_size_y = {{ workgroup_size_y }}u;
    let wg = workgroup_id.x * num_workgroups.y + workgroup_id.y;
    let n_threads_per_workgroup = wg_size_x * wg_size_y;
    let wg_offset = wg * n_threads_per_workgroup;
    let unique_thread_id = wg_offset + local_id;
    let large_prime = 1000000007u;
    let thread_seed = large_prime * unique_thread_id;
    
    var state: array<u32, 4u>;
    for (var i = 0u; i < 4u; i++) {
        state[i] = info[i + 1u] + thread_seed;
    }

    let n_values_per_thread = info[0u];
    for (var i = 0u; i < n_values_per_thread; i++) {
        state[0u] = taus_step(state[0u], 13u, 19u, 12u, 4294967294u);
        state[1u] = taus_step(state[1u], 2u, 25u, 4u, 4294967288u);
        state[2u] = taus_step(state[2u], 3u, 11u, 17u, 4294967280u);
        state[3u] = lcg_step(state[3u]);
        let hybrid_taus = state[0u] ^ state[1u] ^ state[2u] ^ state[3u];
        let write_index = wg_offset * n_values_per_thread + local_id + i * n_threads_per_workgroup;
        let float = cast_float(hybrid_taus);

        let low = args[0];
        let high = args[1];
        let scale = high - low;
        let bias = low;
        output[write_index] = float * scale + bias;
    }
}

fn lcg_step(z: u32) -> u32 {
    return (1664525u * z + 1013904223u);// % 4294967296u; // modulo 2^32, not necessary in u32
}

fn taus_step(z: u32, s1: u32, s2: u32, s3: u32, m: u32) -> u32 {   
    let b = ((z << s1) ^ z) >> s2;  
    return (z & m) << s3 ^ b; 
} 

fn cast_float(number: u32) -> {{ elem }} {
   return 2.3283064365387e-10 * {{ elem }}(number);
}