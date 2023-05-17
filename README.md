# SIMD

`simd_unroller.hh` is a templated function that will automatically unroll and reorder SIMD-operations repeated on an arbitrary length array. An example is in `doubler`.

`simd_guide.hh` is an attempt to build something like Eduardo's SWAR library, where we restrict scarlar code to SIMD-friendly operations, which should enable the compiler to autovectorize what we need. It's not working out as well as planned at the moment.