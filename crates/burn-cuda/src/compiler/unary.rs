use super::{Component, Elem, Item, Variable};
use std::fmt::Display;

pub trait Unary {
    fn format(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable,
        out: &Variable,
    ) -> std::fmt::Result {
        match out.item() {
            Item::Vec4(elem) => Self::format_vec4(f, input, out, elem),
            Item::Vec3(elem) => Self::format_vec3(f, input, out, elem),
            Item::Vec2(elem) => Self::format_vec2(f, input, out, elem),
            Item::Scalar(elem) => Self::format_scalar(f, *input, *out, elem),
        }
    }

    fn format_scalar<Input, Out>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        out: Out,
        elem: Elem,
    ) -> std::fmt::Result
    where
        Input: Component,
        Out: Component;

    fn format_vec2(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        let input0 = input.index(0);
        let input1 = input.index(1);

        let out0 = out.index(0);
        let out1 = out.index(1);

        Self::format_scalar(f, input0, out0, elem)?;
        Self::format_scalar(f, input1, out1, elem)?;

        Ok(())
    }

    fn format_vec3(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        let input0 = input.index(0);
        let input1 = input.index(1);
        let input2 = input.index(2);

        let out0 = out.index(0);
        let out1 = out.index(1);
        let out2 = out.index(2);

        Self::format_scalar(f, input0, out0, elem)?;
        Self::format_scalar(f, input1, out1, elem)?;
        Self::format_scalar(f, input2, out2, elem)?;

        Ok(())
    }

    fn format_vec4(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        let input0 = input.index(0);
        let input1 = input.index(1);
        let input2 = input.index(2);
        let input3 = input.index(3);

        let out0 = out.index(0);
        let out1 = out.index(1);
        let out2 = out.index(3);
        let out3 = out.index(2);

        Self::format_scalar(f, input0, out0, elem)?;
        Self::format_scalar(f, input1, out1, elem)?;
        Self::format_scalar(f, input2, out2, elem)?;
        Self::format_scalar(f, input3, out3, elem)?;

        Ok(())
    }
}

macro_rules! function {
    ($name:ident, $func:expr) => {
        pub struct $name;

        impl Unary for $name {
            fn format_scalar<Input: Display, Out: Display>(
                f: &mut std::fmt::Formatter<'_>,
                input: Input,
                out: Out,
                _elem: Elem,
            ) -> std::fmt::Result {
                f.write_fmt(format_args!("{out} = {}({input});\n", $func))
            }
        }
    };
}

function!(Erf, "erff");
function!(Log, "log");
function!(Exp, "exp");

pub struct Assign;

impl Unary for Assign {
    fn format_scalar<Input, Out>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        out: Out,
        elem: Elem,
    ) -> std::fmt::Result
    where
        Input: Component,
        Out: Component,
    {
        // Cast only when necessary.
        if elem != input.elem() {
            f.write_fmt(format_args!("{out} = {elem}({input});\n"))
        } else {
            f.write_fmt(format_args!("{out} = {input};\n"))
        }
    }
}
