use super::Variable;

macro_rules! gpu {
    // out = lhs + rhs
    ($scope:expr, $out:ident = $lhs:ident + $rhs:expr) => {
        gpu!($scope, $out = add($lhs, $rhs))
    };
    // out = add(lhs, rhs)
    ($scope:expr, $out:ident = add($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Add(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs - rhs
    ($scope:expr, $out:ident = $lhs:ident - $rhs:expr) => {
        gpu!($scope, $out = sub($lhs, $rhs))
    };
    // out = sub(lhs, rhs)
    ($scope:expr, $out:ident = sub($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Sub(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs * rhs
    ($scope:expr, $out:ident = $lhs:ident * $rhs:expr) => {
        gpu!($scope, $out = mul($lhs, $rhs))
    };
    // out = mul(lhs, rhs)
    ($scope:expr, $out:ident = mul($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Mul(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs / rhs
    ($scope:expr, $out:ident = $lhs:ident / $rhs:expr) => {
        gpu!($scope, $out = div($lhs, $rhs))
    };
    // out = div(lhs, rhs)
    ($scope:expr, $out:ident = div($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Div(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs % rhs
    ($scope:expr, $out:ident = $lhs:ident % $rhs:expr) => {
        gpu!($scope, $out = modulo($lhs, $rhs))
    };
    // out = modulo(lhs, rhs)
    ($scope:expr, $out:ident = modulo($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Modulo(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs ^ rhs
    ($scope:expr, $out:ident = $lhs:ident ^ $rhs:expr) => {
        gpu!($scope, $out = powf($lhs, $rhs))
    };
    // out = powf(lhs, rhs)
    ($scope:expr, $out:ident = powf($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Powf(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs == rhs
    ($scope:expr, $out:ident = $lhs:ident == $rhs:expr) => {
        gpu!($scope, $out = equal($lhs, $rhs))
    };
    // out = equal(lhs, rhs)
    ($scope:expr, $out:ident = equal($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Equal(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs > rhs
    ($scope:expr, $out:ident = $lhs:ident > $rhs:expr) => {
        gpu!($scope, $out = greater($lhs, $rhs))
    };
    // out = greater(lhs, rhs)
    ($scope:expr, $out:ident = greater($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Greater(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs >= rhs
    ($scope:expr, $out:ident = $lhs:ident >= $rhs:expr) => {
        gpu!($scope, $out = greater_equal($lhs, $rhs))
    };
    // out = greater_equal(lhs, rhs)
    ($scope:expr, $out:ident = greater_equal($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::GreaterEqual(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs < rhs
    ($scope:expr, $out:ident = $lhs:ident < $rhs:expr) => {
        gpu!($scope, $out = lower($lhs, $rhs))
    };
    // out = lower(lhs, rhs)
    ($scope:expr, $out:ident = lower($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Lower(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs <= rhs
    ($scope:expr, $out:ident = $lhs:ident <= $rhs:expr) => {
        gpu!($scope, $out = lower_equal($lhs, $rhs))
    };
    // out = lower_equal(lhs, rhs)
    ($scope:expr, $out:ident = lower_equal($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::LowerEqual(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = lhs[rhs]
    ($scope:expr, $out:ident = $lhs:ident[$rhs:expr]) => {
        gpu!($scope, $out = index($lhs, $rhs))
    };
    // out = index(lhs, rhs)
    ($scope:expr, $out:ident = index($lhs:expr, $rhs:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Index(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out[lhs] = rhs
    ($scope:expr, $out:ident[$lhs:ident] = $rhs:expr) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::IndexAssign(
            gpu!(binary $lhs, $rhs, $out)
        ));
    };
    // out = |input|
    ($scope:expr, $out:ident = |$input:ident|) => {
        gpu!($scope, $out = abs($input))
    };
    // out = abs(input)
    ($scope:expr, $out:ident = abs($input:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Abs(
            gpu!(unary $input, $out)
        ));
    };
    // out = exp(input)
    ($scope:expr, $out:ident = exp($input:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Exp(
            gpu!(unary $input, $out)
        ));
    };
    // out = log(input)
    ($scope:expr, $out:ident = log($input:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Log(
            gpu!(unary $input, $out)
        ));
    };
    // out = log1p(input)
    ($scope:expr, $out:ident = log1p($input:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Log1p(
            gpu!(unary $input, $out)
        ));
    };
    // out = cos(input)
    ($scope:expr, $out:ident = cos($input:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Cos(
            gpu!(unary $input, $out)
        ));
    };
    // out = sin(input)
    ($scope:expr, $out:ident = sin($input:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Sin(
            gpu!(unary $input, $out)
        ));
    };
    // out = tanh(input)
    ($scope:expr, $out:ident = tanh($input:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Tanh(
            gpu!(unary $input, $out)
        ));
    };
    // out = sqrt(input)
    ($scope:expr, $out:ident = sqrt($input:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Sqrt(
            gpu!(unary $input, $out)
        ));
    };
    // out = erf(input)
    ($scope:expr, $out:ident = erf($input:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::Erf(
            gpu!(unary $input, $out)
        ));
    };
    // out = input
    ($scope:expr, eval $arg:expr) => {
        gpu!($scope, $arg);
    };
    // out = input
    ($scope:expr, $out:ident = $input:ident) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::AssignLocal(
            gpu!(unary $input, $out)
        ));
    };
    // out = input
    ($scope:expr, $out:ident = $input:ident) => {
        $scope.register($crate::codegen::dialect::gpu::Operator::AssignLocal(
            gpu!(unary $input, $out)
        ));
    };
    ($scope:expr, $out:ident = shape($input:expr, $dim:expr)) => {
        $scope.register($crate::codegen::dialect::gpu::Metadata::Shape {
            dim: $dim,
            var: $input,
            out: $out,
        });
    };
    ($scope:expr, $out:ident = stride($input:expr, $dim:expr)) => {
        $scope.register(Metadata::Stride {
            dim: $dim,
            var: $input,
            out: $out,
        });
    };
    ($scope:expr, range($start:expr, $end:expr).for_each($arg:expr)) => {
        $crate::codegen::dialect::gpu::RangeLoop::register($scope, $start.into(), $end.into(), $arg);
    };
    (binary $lhs:expr, $rhs:expr, $out:expr) => {
        $crate::codegen::dialect::gpu::BinaryOperator {
            lhs: $lhs.into(),
            rhs: $rhs.into(),
            out: $out.into(),
        }
    };
    (unary $input:expr, $out:expr) => {
        $crate::codegen::dialect::gpu::UnaryOperator {
            input: $input.into(),
            out: $out.into(),
        }
    };
}

impl From<bool> for Variable {
    fn from(value: bool) -> Self {
        Self::ConstantScalar(if value { 1.0 } else { 0.0 }, super::Elem::Bool)
    }
}

impl From<i32> for Variable {
    fn from(value: i32) -> Self {
        Self::ConstantScalar(value as f64, super::Elem::Int)
    }
}

impl From<f32> for Variable {
    fn from(value: f32) -> Self {
        Self::ConstantScalar(value as f64, super::Elem::Float)
    }
}

impl From<u32> for Variable {
    fn from(value: u32) -> Self {
        Self::ConstantScalar(value as f64, super::Elem::UInt)
    }
}

impl From<usize> for Variable {
    fn from(value: usize) -> Self {
        Self::ConstantScalar(value as f64, super::Elem::UInt)
    }
}

pub(crate) use gpu;
