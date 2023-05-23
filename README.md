## new SIMD library mockup

The biggest problem I can tell with `std::simd` is that it's trying to be both a container and a wrapper around intrinsics, and it's ultimately not being very good at either. I propose instead to break apart the two. For the wrapper, we can encapsulate two different topics. We can simplify the function calls, and hide from the user the need to match the exact code they're writing to the architecture it will be built on. In `intrinsic_generator.hh`, I did these two things on a very limited scope of architectures and functionality.

`simd_container.hh` contains a class called `simd_view`, which is currently my leading choice for how to apply the intrinsic wrapper functions. Within the `simd_view` class, there's a method called `process`.  This takes an instance of `UnrollerUnit` class. `UnrollerUnit` contains all of the information we need to apply a transformation of the data on arbitrary length arrays. `std_simd_proposed_unroller.hh` has a template template function called `unroller` that makes up for the problem that compilers don't optimize very well in the presence of intrinsics. The information contained in `UnrollerUnit` is everything `unroller` needs to efficiently apply intriniscs.  `UnrollerUnit` is a base class using the CRTP. It has default implementations, but some or all of these need to be overridden depending on the operation you want applied to the `simd_view`.


## Examples
- `doubler` doubles an input array `x` and assigns it to an output array `y`.
- `min_finder` finds the minimum element of an array and returns its value.