// This module contains tests to verify substituded functions are the same with no_std and std

#[cfg(test)]
mod tests {

    #[test]
    fn ceil_test() {
        assert_eq!(libm::ceilf(1.1), f32::ceil(1.1)); // 2.0
        assert_eq!(libm::ceilf(2.9), f32::ceil(2.9)); // 3.0

        assert_eq!(libm::ceil(1.1), f64::ceil(1.1)); // 2.0
        assert_eq!(libm::ceil(2.9), f64::ceil(2.9)); // 3.0
    }

    #[test]
    fn round_check() {
        assert_eq!(libm::roundf(-1.0), f32::round(-1.0)); //-1.0
        assert_eq!(libm::roundf(2.8), f32::round(2.8)); //3.0
        assert_eq!(libm::roundf(-0.5), f32::round(-0.5)); //-1.0
        assert_eq!(libm::roundf(0.5), f32::round(0.5)); //1.0
        assert_eq!(libm::roundf(-1.5), f32::round(-1.5)); //-2.0
        assert_eq!(libm::roundf(1.5), f32::round(1.5)); //2.0

        assert_eq!(libm::round(-1.0), f64::round(-1.0)); //-1.0
        assert_eq!(libm::round(2.8), f64::round(2.8)); //3.0
        assert_eq!(libm::round(-0.5), f64::round(-0.5)); //-1.0
        assert_eq!(libm::round(0.5), f64::round(0.5)); //1.0
        assert_eq!(libm::round(-1.5), f64::round(-1.5)); //-2.0
        assert_eq!(libm::round(1.5), f64::round(1.5)); //2.0
    }

    #[test]
    fn sqrt_check() {
        assert_eq!(libm::sqrtf(100.0), f32::sqrt(100.0)); // 10.0
        assert_eq!(libm::sqrtf(4.0), f32::sqrt(4.0)); //2.0);

        assert_eq!(libm::sqrt(100.0), f64::sqrt(100.0)); // 10.0
        assert_eq!(libm::sqrt(4.0), f64::sqrt(4.0)); //2.0);
    }
    #[test]
    fn pow_check() {
        assert_eq!(libm::powf(2.0, 20.0), f32::powf(2.0, 20.0)); // (1 << 20)
        assert_eq!(libm::powf(-1.0, 9.0), f32::powf(-1.0, 9.0)); // -1.0
        assert_eq!(
            libm::powf(-1.0, 2.2).is_nan(),
            f32::powf(-1.0, 2.2).is_nan()
        );
        assert_eq!(
            libm::powf(-1.0, -1.14).is_nan(),
            f32::powf(-1.0, -1.14).is_nan()
        );
    }

    #[test]
    fn log_check() {
        assert_eq!(libm::logf(0.2), f32::ln(0.2));
        assert_eq!(libm::logf(2.0), f32::ln(2.0));
        assert_eq!(libm::logf(-0.2).is_nan(), f32::ln(-0.2).is_nan());
        assert_eq!(libm::logf(0.0).is_nan(), f32::ln(0.0).is_nan());

        assert_eq!(libm::log(0.2), f64::ln(0.2));
        assert_eq!(libm::log(2.0), f64::ln(2.0));
        assert_eq!(libm::log(-0.2).is_nan(), f64::ln(-0.2).is_nan());
        assert_eq!(libm::log(0.0).is_nan(), f64::ln(0.0).is_nan());
    }

    #[test]
    fn log1p_check() {
        assert_eq!(libm::log1pf(0.2), f32::ln_1p(0.2));
        assert_eq!(libm::log1pf(2.0), f32::ln_1p(2.0));
        assert_eq!(libm::log1pf(-0.2).is_nan(), f32::ln_1p(-0.2).is_nan());
        assert_eq!(libm::log1pf(0.0).is_nan(), f32::ln_1p(0.0).is_nan());

        assert_eq!(libm::log1p(0.2), f64::ln_1p(0.2));
        assert_eq!(libm::log1p(2.0), f64::ln_1p(2.0));
        assert_eq!(libm::log1p(-0.2).is_nan(), f64::ln_1p(-0.2).is_nan());
        assert_eq!(libm::log1p(0.0).is_nan(), f64::ln_1p(0.0).is_nan());
    }



    #[test]
    fn exp_check() {
        assert_eq!(libm::expf(0.2), f32::exp(0.2));
        assert_eq!(libm::expf(2.0), f32::exp(2.0));
        assert_eq!(libm::expf(-0.2).is_nan(), f32::exp(-0.2).is_nan());
        assert_eq!(libm::expf(0.0).is_nan(), f32::exp(0.0).is_nan());

        assert_eq!(libm::exp(0.2), f64::exp(0.2));
        assert_eq!(libm::exp(2.0), f64::exp(2.0));
        assert_eq!(libm::exp(-0.2).is_nan(), f64::exp(-0.2).is_nan());
        assert_eq!(libm::exp(0.0).is_nan(), f64::exp(0.0).is_nan());
    }    
}
