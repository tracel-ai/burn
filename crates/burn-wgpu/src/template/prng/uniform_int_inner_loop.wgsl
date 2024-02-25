let low = u32(args[0]);
let high = u32(args[1]);
let range = high - low;

let safe_range = max(range, 1u); // Ensure range is not zero to avoid division by 0 in % op

for (var i = 0u; i < n_values_per_thread; i++) {
    let write_index = write_index_base + i * n_threads_per_workgroup;

    state[0u] = taus_step_0(state[0u]);
    state[1u] = taus_step_1(state[1u]);
    state[2u] = taus_step_2(state[2u]);
    state[3u] = lcg_step(state[3u]);
    let random_u32 = state[0u] ^ state[1u] ^ state[2u] ^ state[3u];

    // Modulus operation to fit within the range
    let mod_result: u32 = u32(random_u32 % safe_range);
    let add_result: u32 = u32(mod_result + low);

    output[write_index] = add_result;
}
