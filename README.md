# simd_unroller.hh

`simd_unroller.hh` is a templated function that will automatically unroll and reorder SIMD operations repeated on an arbitrary length array. There are several usage examples of this. 

- `doubler` doubles an input array `x` and assigns it to an output array `y`.
- `min_finder` finds the minimum element of an array and returns its value.
- `agner_exp` uses an external library (agner fog's vcl) to take the exponential of an input array `x` and assign the output to `y`

### simd_guide.hh

`simd_guide.hh` is an attempt to build something like Eduardo's SWAR library, where we restrict scarlar code to SIMD-friendly operations, which should enable the compiler to autovectorize what we need. It's not working out as well as planned at the moment.