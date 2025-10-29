// Minimal Zig GEMM implementation with C ABI
// Compile with: zig cc -O ReleaseSmall -fPIC -shared -o libgemm.so native/gemm.zig

pub export fn gemm_f32(a: [*]const f32, b: [*]const f32, c: [*]f32, m: usize, n: usize, k: usize) void {
    // SIMD-friendly blocked GEMM.
    // Primary strategy: iterate outer over M, then N in small vector-width blocks (JN)
    // so the inner K loop multiplies a scalar from A by a contiguous vector slice of B.
    // This layout is auto-vectorizer-friendly and gives a clear fast path for future
    // explicit SIMD intrinsics. We unroll the K loop to improve ILP.

    const BM: usize = 64; // block size for M
    const BN: usize = 64; // block size for N
    const BK: usize = 64; // block size for K
    // Choose an inner N block width that maps well to common SIMD widths:
    // - AArch64/NEON: 4 lanes of f32
    // - x86_64/AVX(2): 8 lanes of f32
    // Allow the compiler to select the best codegen for the host at compile-time.
    const JN: usize = comptime if (@import("builtin").arch == .aarch64) 4 else 8;

    var ii0: usize = 0;
    while (ii0 < m) : (ii0 += BM) {
        const i_max = if (ii0 + BM < m) ii0 + BM else m;
        var jj0: usize = 0;
        while (jj0 < n) : (jj0 += BN) {
            const j_max = if (jj0 + BN < n) jj0 + BN else n;
            var pp0: usize = 0;
            while (pp0 < k) : (pp0 += BK) {
                const p_max = if (pp0 + BK < k) pp0 + BK else k;

                var ii: usize = ii0;
                while (ii < i_max) : (ii += 1) {
                    const a_row = a + ii * k;
                    var jj: usize = jj0;
                    while (jj < j_max) : (jj += JN) {
                        // handle a block of up to JN columns at once
                        const actual_j = if (jj + JN < j_max) JN else j_max - jj;

                        // initialize accumulators for the JN lanes
                        var acc: [JN]f32 = undefined;
                        var lane: usize = 0;
                        while (lane < actual_j) : (lane += 1) {
                            acc[lane] = 0.0;
                        }

                        var pp: usize = pp0;

                        // Unrolled K loop to expose ILP and help auto-vectorizers
                        while (pp + 4 <= p_max) : (pp += 4) {
                            // pp + 0
                            const a0: f32 = a_row[pp + 0];
                            var l: usize = 0;
                            while (l < actual_j) : (l += 1) {
                                acc[l] += a0 * b[(pp + 0) * n + (jj + l)];
                            }

                            // pp + 1
                            const a1: f32 = a_row[pp + 1];
                            l = 0;
                            while (l < actual_j) : (l += 1) {
                                acc[l] += a1 * b[(pp + 1) * n + (jj + l)];
                            }

                            // pp + 2
                            const a2: f32 = a_row[pp + 2];
                            l = 0;
                            while (l < actual_j) : (l += 1) {
                                acc[l] += a2 * b[(pp + 2) * n + (jj + l)];
                            }

                            // pp + 3
                            const a3: f32 = a_row[pp + 3];
                            l = 0;
                            while (l < actual_j) : (l += 1) {
                                acc[l] += a3 * b[(pp + 3) * n + (jj + l)];
                            }
                        }

                        // Remainder K
                        while (pp < p_max) : (pp += 1) {
                            const av = a_row[pp];
                            var l2: usize = 0;
                            while (l2 < actual_j) : (l2 += 1) {
                                acc[l2] += av * b[pp * n + (jj + l2)];
                            }
                        }

                        // Write accumulators back to C
                        var out_lane: usize = 0;
                        while (out_lane < actual_j) : (out_lane += 1) {
                            const idx = ii * n + (jj + out_lane);
                            c[idx] = c[idx] + acc[out_lane];
                        }
                    }
                }
            }
        }
    }
}
