# random-variate-poisson

A fast way of generating integer random variates from the Poisson distribution with given lamba, optimized for using vector processing instructions on x86-64 and ARM

# Overview

This library contains a single function that allows very fast generation of random Poisson variates with a given lambda ($\lambda$). It:

* is written in C (although builds cleanly with C++, and the tests are written in C++),

* is very fast, using SSE on x86-64 and NEON on ARMv8-A,

* only uses integer instructions so will be consistent across platform,

* has been tested on x86-64 with gcc/g++/clang, ARMv8-A with clang and x86-64 with Visual C++.

# Use cases

This is useful where a large number of random Poisson variates must be generated where a close approximation is sufficent.

This could be very useful for determining how many objects to place in games or simulations. For instance in a galaxy simulator, it can be used to determine the number of stars in a voxel.

# Usage

There is a single function:

	uint32_t poisson_random_variable_fixed_int(uint64_t* seed, int64_t lambda);

where:

* **seed** is a pointer to the random seed (which will be modified),

* **lambda** is the Poisson parameter $\lambda$ was a 32.32 fixed point integer,

* the random variate is returned.

If a **float** or **double** is desired, just multiply by 4294967296ULL (0x100000000ULL).

	poisson_random_variable_fixed_int(&seed,lambda*4294967296ULL);

### Files

**RandomVariatePoisson.h**: file to include to access the C functionality. Some of the basic functions are provided by inline functions.

**RandomVariatePoisson.c**: Add this into your project. This includes the functionality not provided by pseudo_double.h

**PoissonTest.cpp**: Some simple tests

# Design Considerations

The basic method (in pseudocode):

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

Now note that this can be done on 32 or 64 bit integers, but can be adjusted to work on integers as small as 8 bits (in practice, there will need to be some modifications to prevent the random number being 0, and to deal with rounding on arithmetic with very small values).

This can now be vectorized - both SSE and NEON support vectors of 16 8 bit unsigned integers, so 16 versions of the loop can be run at once.

Note that with AVX2 or AVX512, 32 or 64 byte vectors could be used, but we choose to stick to 16 bytes to maintain cross platform consistency.
