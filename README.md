# simd_unroller.hh

`simd_unroller.hh` is a templated function that will automatically unroll and reorder SIMD operations repeated on an arbitrary length array. There are several usage examples of this. 

- `doubler` doubles an input array `x` and assigns it to an output array `y`.
- `min_finder` finds the minimum element of an array and returns its value.
- `agner_exp` uses an external library (agner fog's vcl) to take the exponential of an input array `x` and assign the output to `y`.
- `stl_halve` uses the STL's experimental simd library to halve values from an array `x` and assigns it to output `y`.

This unroller lacks a lot. It currently accepts one arithmetic input array and maps it to one arithmetic output array, though the output array can either be interpreted as size N or size 1 (for reduce operations). This should be generalized to M inputs mapped to N outputs. 

I also do not know how to abstract this away from the user. Maybe put inside a `simd_container` type where the user can call `process(lambda)`?

### simd_guide.hh

`simd_guide.hh` is an attempt to build something like Eduardo's SWAR library, where we restrict scalar code to SIMD-friendly operations, which should enable the compiler to autovectorize what we need. It's not working out as well as planned at the moment.