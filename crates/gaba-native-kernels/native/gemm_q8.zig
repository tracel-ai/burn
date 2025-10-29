// Simple quantized GEMM: inputs are u8, outputs are i64 accumulators per element
// ABI: gemm_q8_to_i64(a: [*]const u8, b: [*]const u8, c: [*]i64, m: usize, n: usize, k: usize)

pub export fn gemm_q8_to_i64(a: [*]const u8, b: [*]const u8, c: [*]i64, m: usize, n: usize, k: usize) void {
    // Zero output
    var idx: usize = 0;
    while (idx < m * n) : (idx += 1) {
        c[idx] = 0;
    }

    const BM: usize = 32;
    const BN: usize = 32;
    const BK: usize = 32;

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
                    var jj: usize = jj0;
                    while (jj < j_max) : (jj += 1) {
                        var sum: i64 = 0;
                        var pp: usize = pp0;
                        while (pp < p_max) : (pp += 1) {
                            sum += (@as(i64, a[ii * k + pp])) * (@as(i64, b[pp * n + jj]));
                        }
                        const idx2 = ii * n + jj;
                        c[idx2] = c[idx2] + sum;
                    }
                }
            }
        }
    }
}
