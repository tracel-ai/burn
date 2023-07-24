/// Metal has a weird numerical behaviour with tanh for inputs over 43.0
fn safe_tanh(x: {{ elem }}) -> {{ elem }} {
    if x > 43.0 {
        return 1.0;
    } else {
        return tanh(x);
    }
}
