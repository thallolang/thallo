Thallo
---

Thallo (https://www.thallo-lang.org/) is a new version of Opt (optlang.org), a new language in which a user simply writes energy functions over image- or graph-structured unknowns, and a compiler automatically generates state-of-the-art GPU optimization kernels. Real-world energy functions compile directly into highly optimized GPU solver implementations with performance competitive with the best published hand-tuned, application-specific GPU solvers.

Thallo introduces *scheduling* to this problem domain, as discussed in the accompanying publication: [Thallo – Scheduling for High-Performance Large-scale Non-linear Least-Squares Solvers](https://light.cs.princeton.edu/wp-content/uploads/2021/06/THALLO.pdf)

This is a pre-alpha release of the software. An alpha release, with reproducible build instructions and scripts, docker images, and documentation is now slated for a September 2021 release.

As a pre-alpha release, there are no guarantees made; the alpha release will seek to support all popular runtime environments for modern Cuda (such as cloud V100 machines); it is known that some combinations of versions of LLVM/Terra/Cuda/GPUs/Driver Versions produce incorrect PTX; this will be resolved and fully documented in the alpha release.

Open an issue on (https://github.com/thallolang/thallo) if you see improvements that could be made that are not covered by the above. 


### Prerequisites ###

Overview
========

Thallo is composed of a library `libThallo.a` and a header file `Thallo.h`. An application links Thallo and uses its API to define and solve optimization problems. Thallo's high-level energy functions behave like shaders in OpenGL. They are loaded as your application runs using the `Thallo_ProblemDefine` API.

See the Makefiles in the examples for instructions on how to link Thallo into your applications. 

