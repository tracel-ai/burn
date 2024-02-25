fn box_muller_transform(unit_1: {{ elem }}, unit_2: {{ elem }}) -> array<{{ elem }}, 2> {
    let mean = args[0];
    let stdev = args[1];
    let coeff = stdev * sqrt(-2.0 * log(unit_1));
    
    let pi = 3.141592653589793238;
    let trigo_arg = 2.0 * pi * unit_2;
    let cos_ = cos(trigo_arg);
    let sin_ = sin(trigo_arg);
    
    return array(coeff * cos_ + mean, coeff * sin_ + mean);
}