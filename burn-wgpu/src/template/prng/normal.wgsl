for (var i = 0u; i < n_values_per_thread / 2u; i++) {
    let write_index_0 = write_index_base + (2u * i) * n_threads_per_workgroup;
    let write_index_1 = write_index_0 + n_threads_per_workgroup;
    
    state[0u] = taus_step_0(state[0u]);
    state[1u] = taus_step_1(state[1u]);
    state[2u] = taus_step_2(state[2u]);
    state[3u] = lcg_step(state[3u]);
    let random_1_u32 = state[0u] ^ state[1u] ^ state[2u] ^ state[3u];
    let random_1 = cast_float(random_1_u32);

    state[0u] = taus_step_0(state[0u]);
    state[1u] = taus_step_1(state[1u]);
    state[2u] = taus_step_2(state[2u]);
    state[3u] = lcg_step(state[3u]);
    let random_2_u32 = state[0u] ^ state[1u] ^ state[2u] ^ state[3u];
    let random_2 = cast_float(random_2_u32);

    let transformed = box_muller_transform(random_1, random_2);

    output[write_index_0] = transformed[0];
    output[write_index_1] = transformed[1];
}