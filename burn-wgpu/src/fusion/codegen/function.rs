use super::Elem;
use std::fmt::Display;

#[derive(Hash, PartialEq, Eq, Clone)]
pub enum Function {
    Powf(Elem),
}

impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Function::Powf(elem) => f.write_fmt(format_args!(
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
            )),
        }
    }
}
