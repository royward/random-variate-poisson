# random-variate-poisson

A fast way of generating integer random variates from the Poisson distribution with given lamba, optimized for using vector processing instructions on x86-64 and ARM

# Overview

This library contains three variations of functions that allows fast generation of random Poisson variates with a given lambda ($\lambda$). It:

* is written in C (although builds cleanly with C++, and the tests are written in C++),

* is very fast, using SSE on x86-64 and NEON on ARMv8-A,

* only uses integer instructions so will be consistent across platform (for two of the versions),

* has been tested on x86-64 with gcc/g++/clang, ARMv8-A with clang and x86-64 with Visual C++.

# Use cases

This is useful where a large number of random Poisson variates must be generated, particularly when $\lambda$ varies across invocations.

The `poisson_random_variate_integer` variations only use integer operations, so will be consistent cross-platform. This could be very useful for reproducibly determining how many objects to place in games or simulations. For instance in a galaxy simulator, it can be used to determine the number of stars in a voxel. This might also be of use on platforms where floating point support is not available.

# Usage

There are two functions:

	uint32_t poisson_random_variate_integer(uint64_t* seed, int64_t lambda);

	uint32_t poisson_random_variate_double(uint64_t* seed, double lambda);

where:

* **seed** is a pointer to the random seed (which will be modified),

* **lambda** is the Poisson parameter. $\lambda$ was a 32.32 fixed point integer in the integer case, a `double` in the double case,

* the random variate is returned.

If a **float** or **double** is desired in the integer version, just multiply by 4294967296ULL (0x100000000ULL).

	poisson_random_variate_integer(&seed,lambda*4294967296ULL);

### Files

**poisson_random_variate_integer.h** file to include to access the C functionality for `poisson_random_variate_integer`

**poisson_random_variate_integer.c** the implementation of `poisson_random_variate_integer`

**poisson_random_variate_old.c** an old implementation of poisson_random_variate_integer. This is slower than the more recent version, slightly buggy (the means are correct, but the distributions are narrower than they should be). Don't use this unless you have been already using this and need the exact results used by 1.0.0.

**poisson_random_variate_double.h** file to include to access the C functionality for `poisson_random_variate_double`

**poisson_random_variate_double.c** the implementation of `poisson_random_variate_double`

**PoissonTest.cpp**: Some simple tests

# Design Considerations

For large $\lambda$, use PTRD algorithm described by Wolfgang H&ouml;rmann in [The transformed rejection method for generating Poisson random variables](https://www.sciencedirect.com/science/article/abs/pii/0167668793909974) in Insurance: Mathematics and Economics, Volume 12, Issue 1, February 1993, Pages 39-45. A non pay-walled version is [here](https://research.wu.ac.at/ws/portalfiles/portal/18953249/document.pdf).

For small $\lambda$, the basic methods (in pseudocode):

	poisson_random_variable(lambda)
		let l=exp(-lambda)
		let p=1
		let k=-1
		do
			k=k+1
			p=p*(random number in [0,1])
		while
			p>l
		return k

This has a number of issues:

* exp(-lambda) will underflow for large lambda

* will have to execute the loop an average of lambda times

The first step to speeding this up is to covert to unsigned integers and use a count leading zeros (clz) function to keep renormalizing:

	poisson_random_variable(lambda)
		let num_digits=lambda/log(2.0)
		let int_digits=floor(num_digits)
		let start=exp(frac(num_digits)*log(2.0))*(1<<(integer bit size-1))
		let k=0
		while int_digits>0 do
			k=k+1
			start=mult_hi_part(startx,(random number in [0,max unsigned int]
			let z=clz(start)
			int_digits-=z
			start=start<<z
		return k

Now note that this can be done on 32 or 64 bit integers, but can be adjusted to work on integers as small as 16 bits (8 bits have rounding errors).

This can now be vectorized - both SSE and NEON support vectors of 8 16 bit unsigned integers, so 8 versions of the loop can be run at once.

Note that with AVX2 or AVX512, 32 or 64 byte vectors could be used, but we choose to stick to 16 bytes to maintain cross platform consistency.

A full descrption of all this can be found in my blog post: [Fast Integer Poisson Random Variates for Procedural Generation](https://www.orange-kiwi.com/posts/fast-integer-poisson-random-variates-for-procedural-generation/)
