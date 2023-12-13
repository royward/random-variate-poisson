# random-variate-poisson

A fast way of generating integer random variates from the Poisson distribution with given lamba, optimized for using vector processing instructions on x86-64 and ARM

# Overview

This library contains a single function that allows very fast generation of random Poisson variates with a given lambda ($\lambda$). It:

* is written in C (although builds cleanly with C++, and the tests are written in C++),

* is very fast, using SSE on x86-64 and NEON on ARMv8-A,

* only uses integer instructions so will be consistent across platform,

* has been tested on x86-64 with gcc/g++/clang, ARMv8-A with clang and x86-64 with Visual C++.

It also allows the number of expononent bits to be set with a macro, which enables choices to be made in the range/precision tradeoff.

# Use cases

This is useful where a large number of random Poisson variates must be generated where a close approximation is sufficent.

This could be very useful for determining how many objects to place in games or simulations. For instance in a galaxy simulator, it can be used to determine the number of stars in a voxel.

# Usage

There is a single function:

	uint32_t poisson_random_variable_fixed_int(uint64_t* seed, int64_t lambda);

where:

* *seed* is a pointer to the random seed (which will be modified),

* *lambda* is the Poisson parameter $\lambda$ was a 32.32 fixed point integer,

* the random variate is returned.

If a *float* or *double* is desired, just multiply by 4294967296ULL (0x100000000ULL).

	poisson_random_variable_fixed_int(&seed,lambda*4294967296ULL);
