use super::{Component, Elem, InstructionSettings, Item, Variable};
use std::fmt::Display;

pub trait Unary {
    fn format(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable,
        out: &Variable,
    ) -> std::fmt::Result {
        let item = out.item();
        let settings = Self::settings(*item.elem());

        match item {
            Item::Vec4(elem) => {
                if settings.native_vec4 {
                    Self::format_native_vec4(f, input, out, elem)
                } else {
                    Self::unroll_vec4(f, input, out, elem)
                }
            }
            Item::Vec3(elem) => {
                if settings.native_vec3 {
                    Self::format_native_vec3(f, input, out, elem)
                } else {
                    Self::unroll_vec3(f, input, out, elem)
                }
            }
            Item::Vec2(elem) => {
                if settings.native_vec2 {
                    Self::format_native_vec2(f, input, out, elem)
                } else {
                    Self::unroll_vec2(f, input, out, elem)
                }
            }
            Item::Scalar(elem) => Self::format_scalar(f, *input, *out, elem),
        }
    }

    fn settings(_elem: Elem) -> InstructionSettings {
        InstructionSettings::default()
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

    fn format_native_vec4(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        Self::format_scalar(f, *input, *out, elem)
    }

    fn format_native_vec3(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        Self::format_scalar(f, *input, *out, elem)
    }

    fn format_native_vec2(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable,
        out: &Variable,
        elem: Elem,
    ) -> std::fmt::Result {
        Self::format_scalar(f, *input, *out, elem)
    }

    fn unroll_vec2(
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

    fn unroll_vec3(
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

    fn unroll_vec4(
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
        let out2 = out.index(2);
        let out3 = out.index(3);

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

function!(Abs, "abs");
function!(Log, "log");
function!(Log1p, "log1p");
function!(Cos, "cos");
function!(Sin, "sin");
function!(Tanh, "tanh");
function!(Sqrt, "sqrt");
function!(Exp, "exp");
function!(Erf, "erff");
function!(Ceil, "ceil");
function!(Floor, "floor");

pub struct Not;

impl Unary for Not {
    fn format_scalar<Input, Out>(
        f: &mut std::fmt::Formatter<'_>,
        input: Input,
        out: Out,
        _elem: Elem,
    ) -> std::fmt::Result
    where
        Input: Component,
        Out: Component,
    {
        f.write_fmt(format_args!("{out} = !{input};\n"))
    }
}

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
