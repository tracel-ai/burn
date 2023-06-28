fn powf(lhs: {{ elem }}, rhs: {{ elem }}) -> {{ elem }} {
    let modulo = rhs % 2.0;

    if (modulo == 0.0) {
        // Even number
        return pow(abs(lhs), rhs);
    } else if (modulo == 1.0) {
        // Odd number
        return -1.0 * pow(abs(lhs), rhs);
    } else {
        // Float number
        return pow(lhs, rhs);
    }
}
