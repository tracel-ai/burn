use super::Elem;
use std::fmt::Display;

/// Not all functions are native to WGSL, so this struct allows to support more functions.
#[derive(PartialEq, Eq, Clone)]
pub enum Function {
    Powf(Elem),
    Erf(Elem),
    #[cfg(target_os = "macos")]
    SafeTanh(Elem),
}

impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Function::Powf(elem) => format_powf(f, elem),
            Function::Erf(elem) => format_erf(f, elem),
            #[cfg(target_os = "macos")]
            Function::SafeTanh(elem) => format_safe_tanh(f, elem),
        }
    }
}

fn format_powf(f: &mut core::fmt::Formatter<'_>, elem: &Elem) -> core::fmt::Result {
    f.write_fmt(format_args!(
        "
fn powf(lhs: {elem}, rhs: {elem}) -> {elem} {{
    let modulo = rhs % 2.0;

    if (modulo == 0.0) {{
        // Even number
        return pow(abs(lhs), rhs);
    }} else if (modulo == 1.0 && lhs < 0.0) {{
        // Odd number
        return -1.0 * pow(-1.0 * lhs, rhs);
    }} else {{
        // Float number
        return pow(lhs, rhs);
    }}
}}
"
    ))
}

fn format_erf(f: &mut core::fmt::Formatter<'_>, elem: &Elem) -> core::fmt::Result {
    f.write_fmt(format_args!(
        "
/// An approximation of the error function: https://en.wikipedia.org/wiki/Error_function#Numerical_approximations
///
/// > (maximum error: 1.5×10−7)
/// > All of these approximations are valid for x ≥ 0. To use these approximations for negative x, use the fact that erf x is an odd function, so erf x = −erf(−x).
fn erf_positive(x: {elem}) -> {elem} {{
    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;

    let t = 1.0 / (1.0 + p * abs(x));
    let tmp = ((((a5 * t + a4) * t) + a3) * t + a2) * t + a1;

    return 1.0 - (tmp * t * exp(-x * x));
}}

fn erf(x: {elem}) -> {elem} {{
    if (x < 0.0) {{
        return -1.0 * erf_positive(-1.0 * x);
    }}

    return erf_positive(x);
}}
"
    ))
}

#[cfg(target_os = "macos")]
fn format_safe_tanh(f: &mut core::fmt::Formatter<'_>, elem: &Elem) -> core::fmt::Result {
    f.write_fmt(format_args!(
        "
/// Metal has a weird numerical behaviour with tanh for inputs over 43.0
fn safe_tanh(x: {elem}) -> {elem} {{
    if x > 43.0 {{
        return 1.0;
    }} else {{
        return tanh(x);
    }}
}}
"
    ))
}
