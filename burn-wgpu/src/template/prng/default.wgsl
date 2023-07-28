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
    // let global_seed: u32 = info[0];
    // let large_prime = 4u;
    let large_prime = 1000000007u;
    let thread_seed = large_prime * id;

    output[id] = hybrid_taus(thread_seed+info[0],thread_seed+info[1],thread_seed+info[2],thread_seed+info[3]);
}

fn lcg_step(z: u32) -> u32 {
    return (1664525u * z + 1013904223u);// % 4294967296u; // modulo 2^32, not necessary in u32
}

fn taus_step(z: u32, s1: u32, s2: u32, s3: u32, m: u32) -> u32 {   
    let b = ((z << s1) ^ z) >> s2;  
    return (z & m) << s3 ^ b; 
} 

fn hybrid_taus(seed: u32, seed1:u32,seed2:u32,seed3:u32) -> f32 {
    let random_int = taus_step(seed+0u, 13u, 19u, 12u, 4294967294u) ^  
        taus_step(seed1, 2u, 25u, 4u, 4294967288u) ^    
        taus_step(seed2, 3u, 11u, 17u, 4294967280u) ^  
        lcg_step(seed3);
    return 2.3283064365387e-10 * f32(random_int); 
} 