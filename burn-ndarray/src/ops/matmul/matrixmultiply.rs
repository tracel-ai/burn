use super::Gemm;

pub(crate) struct MatrixmultiplyGemm;

impl Gemm<f32> for MatrixmultiplyGemm {
    fn run(
        m: usize,
        k: usize,
        n: usize,
        alpha: f32,
        a: *const f32,
        rsa: isize,
        csa: isize,
        b: *const f32,
        rsb: isize,
        csb: isize,
        beta: f32,
        c: *mut f32,
        rsc: isize,
        csc: isize,
    ) {
        unsafe {
            matrixmultiply::sgemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc);
        }
    }
}

impl Gemm<f64> for MatrixmultiplyGemm {
    fn run(
        m: usize,
        k: usize,
        n: usize,
        alpha: f64,
        a: *const f64,
        rsa: isize,
        csa: isize,
        b: *const f64,
        rsb: isize,
        csb: isize,
        beta: f64,
        c: *mut f64,
        rsc: isize,
        csc: isize,
    ) {
        unsafe {
            matrixmultiply::dgemm(m, k, n, alpha, a, rsa, csa, b, rsb, csb, beta, c, rsc, csc);
        }
    }
}
