use super::Item;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

/// Not all functions are native to WGSL, so this struct allows to support more functions.
#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub enum Function {
    PowfScalar(Item),
    Powf(Item),
    Erf(Item),
    #[cfg(target_os = "macos")]
    SafeTanh(Item),
}

impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Function::PowfScalar(elem) => format_powf_scalar(f, elem),
            Function::Powf(elem) => format_powf(f, elem),
            Function::Erf(elem) => format_erf(f, elem),
            #[cfg(target_os = "macos")]
            Function::SafeTanh(elem) => format_safe_tanh(f, elem),
        }
    }
}

fn format_powf_scalar(f: &mut core::fmt::Formatter<'_>, item: &Item) -> core::fmt::Result {
    base_powf_fmt(f, item)?;

    match item {
        Item::Vec4(elem) => f.write_fmt(format_args!(
            "
fn powf(lhs: {item}, rhs: {elem}) -> {item} {{
    return vec4(
        powf_primitive(lhs[0], rhs),
        powf_primitive(lhs[1], rhs),
        powf_primitive(lhs[2], rhs),
        powf_primitive(lhs[3], rhs),
    );
}}
"
        )),
        Item::Vec3(elem) => f.write_fmt(format_args!(
            "
fn powf(lhs: {item}, rhs: {elem}) -> {item} {{
    return vec3(
        powf_primitive(lhs[0], rhs),
        powf_primitive(lhs[1], rhs),
        powf_primitive(lhs[2], rhs),
    );
}}
"
        )),
        Item::Vec2(elem) => f.write_fmt(format_args!(
            "
fn powf(lhs: {item}, rhs: {elem}) -> {item} {{
    return vec2(
        powf_primitive(lhs[0], rhs),
        powf_primitive(lhs[1], rhs),
    );
}}
"
        )),
        Item::Scalar(elem) => f.write_fmt(format_args!(
            "
fn powf(lhs: {elem}, rhs: {elem}) -> {elem} {{
    return powf_primitive(lhs, rhs);
}}
"
        )),
    }
}

fn base_powf_fmt(f: &mut std::fmt::Formatter<'_>, item: &Item) -> Result<(), std::fmt::Error> {
    let elem = item.elem();
    f.write_fmt(format_args!(
        "
fn powf_primitive(lhs: {elem}, rhs: {elem}) -> {elem} {{
    let modulo = rhs % 2.0;
    if rhs == 0.0 {{
        return 1.0;
    }}
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
    ))?;
    Ok(())
}

fn format_powf(f: &mut core::fmt::Formatter<'_>, item: &Item) -> core::fmt::Result {
    base_powf_fmt(f, item)?;

    match item {
        Item::Vec4(elem) => f.write_fmt(format_args!(
            "
fn powf(lhs: {item}, rhs: {elem}) -> {item} {{
    return vec4(
        powf_primitive(lhs[0], rhs[0]),
        powf_primitive(lhs[1], rhs[1]),
        powf_primitive(lhs[2], rhs[2]),
        powf_primitive(lhs[3], rhs[3]),
    );
}}
"
        )),
        Item::Vec3(elem) => f.write_fmt(format_args!(
            "
fn powf(lhs: {item}, rhs: {elem}) -> {item} {{
    return vec3(
        powf_primitive(lhs[0], rhs[0]),
        powf_primitive(lhs[1], rhs[1]),
        powf_primitive(lhs[2], rhs[2]),
    );
}}
"
        )),
        Item::Vec2(elem) => f.write_fmt(format_args!(
            "
fn powf(lhs: {item}, rhs: {elem}) -> {item} {{
    return vec2(
        powf_primitive(lhs[0], rhs[0]),
        powf_primitive(lhs[1], rhs[1]),
    );
}}
"
        )),
        Item::Scalar(elem) => f.write_fmt(format_args!(
            "
fn powf(lhs: {elem}, rhs: {elem}) -> {elem} {{
    return powf_primitive(lhs, rhs);
}}
"
        )),
    }
}

fn format_erf(f: &mut core::fmt::Formatter<'_>, ty: &Item) -> core::fmt::Result {
    let elem = ty.elem();
    f.write_fmt(format_args!(
        "
/// An approximation of the error function: https://en.wikipedia.org/wiki/Error_function#Numerical_approximations
///
/// > (maximum error: 1.5×10−7)
/// > All of these approximations are valid for x ≥ 0. To use these approximations for negative x, use the fact that erf x is an odd function, so erf x = −erf(−x).
fn erf_positive_scalar(x: {elem}) -> {elem} {{
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

fn erf_scalar(x: {elem}) -> {elem} {{
    if (x < 0.0) {{
        return -1.0 * erf_positive_scalar(-1.0 * x);
    }}

    return erf_positive_scalar(x);
}}
"
    ))?;

    match ty {
        Item::Vec4(_) => f.write_fmt(format_args!(
            "
fn erf(x: {ty}) -> {ty} {{
    return vec4(
       erf_scalar(x[0]),
       erf_scalar(x[1]),
       erf_scalar(x[2]),
       erf_scalar(x[3]),
    );
}}
                "
        )),
        Item::Vec3(_) => f.write_fmt(format_args!(
            "
fn erf(x: {ty}) -> {ty} {{
    return vec3(
       erf_scalar(x[0]),
       erf_scalar(x[1]),
       erf_scalar(x[2]),
    );
}}
                "
        )),
        Item::Vec2(_) => f.write_fmt(format_args!(
            "
fn erf(x: {ty}) -> {ty} {{
    return vec2(
       erf_scalar(x[0]),
       erf_scalar(x[1]),
    );
}}
                "
        )),
        Item::Scalar(_) => f.write_fmt(format_args!(
            "
fn erf(x: {ty}) -> {ty} {{
   return erf_scalar(x);
}}
                "
        )),
    }
}

#[cfg(target_os = "macos")]
fn format_safe_tanh(f: &mut core::fmt::Formatter<'_>, item: &Item) -> core::fmt::Result {
    let elem = item.elem();

    f.write_fmt(format_args!(
        "
/// Metal has a weird numerical behaviour with tanh for inputs over 43.0
fn safe_tanh_scalar(x: {elem}) -> {elem} {{
    if x > 43.0 {{
        return 1.0;
    }} else {{
        return tanh(x);
    }}
}}
"
    ))?;

    match item {
        Item::Vec4(_) => f.write_fmt(format_args!(
            "
fn safe_tanh(x: {item}) -> {item} {{
    return vec4(
        safe_tanh_scalar(x[0]),
        safe_tanh_scalar(x[1]),
        safe_tanh_scalar(x[2]),
        safe_tanh_scalar(x[3]),
    );
}}
"
        )),
        Item::Vec3(_) => f.write_fmt(format_args!(
            "
fn safe_tanh(x: {item}) -> {item} {{
    return vec3(
        safe_tanh_scalar(x[0]),
        safe_tanh_scalar(x[1]),
        safe_tanh_scalar(x[2]),
    );
}}
"
        )),
        Item::Vec2(_) => f.write_fmt(format_args!(
            "
fn safe_tanh(x: {item}) -> {item} {{
    return vec2(
        safe_tanh_scalar(x[0]),
        safe_tanh_scalar(x[1]),
    );
}}
"
        )),
        Item::Scalar(_) => f.write_fmt(format_args!(
            "
fn safe_tanh(x: {item}) -> {item} {{
    return safe_tanh_scalar(x);
}}
"
        )),
    }
}
