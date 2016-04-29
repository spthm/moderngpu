segmentedgpu
============

Segmented parallel primitives for GPUs.

segmentedgpu is a stripped-down version of [moderngpu v1.0](http://nvlabs.github.io/moderngpu) by Sean Baxter. The main differences are as follows:

* Only reductions, scans, sorts and their segmented variants are present (along with any dependencies they may have).
* segmentedgpu does not require separate compilation of a context class; it may be used as a header-only library. However, `CudaContext` is now only accessible from `.cu` files, and so segmentedgpu may only be used in `.cu` files.
* A segmented scan has been added.
* Visual Studio support has been dropped.
* Scans, reductions, sorts and their segmented variants may be run on Fermi-class (compute capability 2.x) hardware without error for large input data sizes. (moderngpu kernel launches assume the required block size does not exceed the device maximum, which in general is true only for 3.x+ devices.) This is achieved with [grid-stride thread loops](https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/).
* A [StreamScan](https://dl.acm.org/citation.cfm?id=2442539) implementation has been added. It requires a CUDA version >= 6.5 for optimal operation. (Performance on prior CUDA versions may be reduced.)
Note that the [current implementation](include/kernels/streamscan.cuh) does not fully conform to the CUDA memory model, and may in principle result in occasional errors (though none have been found on compute capability 2.0, 3.0 and 3.5 devices thus far for CUDA versions 6.5 through 7.5).
* The exponentially-spaced buckets and LRU allocator has been removed in favour of using `cudaMalloc` directly. It is still possible to set a user-provided allocator.
