use super::Gemm;
use cblas_sys::{cblas_dgemm, cblas_sgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE};

pub(crate) struct BlasGemm;

impl Gemm<f32> for BlasGemm {
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
        println!("----");
        println!("csa {csa}");
        println!("csb {csb}");
        println!("csc {csc}");

        let a_trans = match csb > csa {
            true => CBLAS_TRANSPOSE::CblasNoTrans,
            false => CBLAS_TRANSPOSE::CblasNoTrans,
        };
        let b_trans = match csa > csb {
            true => CBLAS_TRANSPOSE::CblasNoTrans,
            false => CBLAS_TRANSPOSE::CblasNoTrans,
        };
        let layout = blas_layout((m, k), (rsa, csa));

        let rsa = i32::max(rsa as i32, k as i32);
        let rsb = i32::max(rsb as i32, n as i32);
        let rsc = i32::max(rsc as i32, n as i32);

        unsafe {
            cblas_sgemm(
                layout, a_trans, b_trans, m as i32, n as i32, k as i32, alpha, a, rsa as i32, b,
                rsb as i32, beta, c, rsc as i32,
            );
        }
    }
}

impl Gemm<f64> for BlasGemm {
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
        println!("----");
        println!("csa {csa}");
        println!("csb {csb}");
        println!("csc {csc}");
        println!("----");

        unsafe {
            cblas_dgemm(
                CBLAS_LAYOUT::CblasRowMajor,
                CBLAS_TRANSPOSE::CblasNoTrans,
                CBLAS_TRANSPOSE::CblasNoTrans,
                m as i32,
                n as i32,
                k as i32,
                alpha,
                a,
                rsa as i32,
                b,
                rsb as i32,
                beta,
                c,
                rsc as i32,
            );
        }
    }
}

fn blas_layout(dim: (usize, usize), strides: (isize, isize)) -> CBLAS_LAYOUT {
    let blas_row_major_2d = is_blas_2d(dim, strides, MemoryOrder::C);

    if blas_row_major_2d {
        return CBLAS_LAYOUT::CblasRowMajor;
    }

    CBLAS_LAYOUT::CblasRowMajor
}

enum MemoryOrder {
    C,
    F,
}

fn is_blas_2d(dim: (usize, usize), stride: (isize, isize), order: MemoryOrder) -> bool {
    let (m, n) = dim;
    let (s0, s1) = stride;
    let (inner_stride, outer_dim) = match order {
        MemoryOrder::C => (s1, n),
        MemoryOrder::F => (s0, m),
    };
    if !(inner_stride == 1 || outer_dim == 1) {
        return false;
    }
    if s0 < 1 || s1 < 1 {
        return false;
    }

    true
}
