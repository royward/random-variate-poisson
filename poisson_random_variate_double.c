// Copyright (c) 2024, Roy Ward
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
// files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy,
// modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <array>

#ifdef _MSC_VER // Windows

static inline void multu64hilo(uint64_t x, uint64_t y, uint64_t* rhi, uint64_t* rlo) {
	*rlo=_umul128(x, y, rhi);
}

#elif defined(__GNUC__) || defined(__clang__) // gcc/clang

static inline void multu64hilo(uint64_t x,uint64_t y,uint64_t* rhi,uint64_t* rlo) {
	unsigned __int128 ret=((unsigned __int128)x)*y;
	*rhi=ret>>64;
	*rlo=ret;
}

#endif

static inline uint64_t fast_rand64(uint64_t* seed) {
	*seed += 0x60bee2bee120fc15ULL;
	uint64_t hi,lo;
	multu64hilo(*seed,0xa3b195354a39b70dULL,&hi,&lo);
	uint64_t m1 = hi^lo;
	multu64hilo(m1,0x1b03738712fad5c9ULL,&hi,&lo);
	uint64_t m2 = hi^lo;
	return m2;
}

double fast_rand_double(uint64_t* seed) {
	// yes, the compiler does the right thing and turns it into a multiplication
	return fast_rand64(seed)/18446744073709551616.0;
}

constexpr int factorial(int n) {
	if(n==0) {
		return 1;
	} else {
		return n*factorial(n-1);
	}
}

constexpr std::array<double,10> log_fact_table = {{
	std::log(factorial(0)),
	std::log(factorial(1)),
	std::log(factorial(2)),
	std::log(factorial(3)),
	std::log(factorial(4)),
	std::log(factorial(5)),
	std::log(factorial(6)),
	std::log(factorial(7)),
	std::log(factorial(8)),
	std::log(factorial(9))
}};

uint32_t poisson_random_variate_double(uint64_t* seed, double lambda) {
	if(lambda<=0) {
		return 0;
	}
	if(lambda<10) {
		uint32_t ret=0;
		while(lambda>0) {
			double l=std::min(lambda,500.0);
			lambda-=l;
			double L=exp(-l);
			double p=fast_rand_double(seed);
			while(p>L) {
				ret++;
				p*=fast_rand_double(seed);
			}
		}
		return ret;
	}
	double u=lambda;
	while(true) {
		double smu=std::sqrt(u); // >=3.1623
		double b=0.931+2.53*smu; // >=8.9316
		double a=-0.059+0.02483*b; // >=0.16277
		double vr=0.9277-3.6224/(b-2.0); // >=0.4051, <=0.9277 
		double V=fast_rand_double(seed);
		if(V<0.86*vr) { // V/vr<0.86
			double U=V/vr-0.43; // >=-0.43, <=0.43
			double us=0.5-abs(U); // >=0.07, <=0.5
			// 2*a/us  >=0.6511 <=4.6501
			// 2*a/us+b >=5.8154
			return std::floor((2.0*a/us+b)*U+u+0.445);
		}
		double t=fast_rand_double(seed);
		double U;
		if(V>=vr) {
			U=t-0.5; // >=-0.5, <=0.5
		} else {
			U=V/vr-0.93; // >=-0.93, <=0.07
			// U<0  => >=-0.5, <=0.43
			// U>=0 => >=0.43 <=0.5
			U=((U<0)?-0.5:0.5)-U; // >=-0.5, <=0.5
			V=t*vr; // >=0, <=0.9277
		}
		// U >=-0.5, <=0.5
		// V >=0, <=1
		double us=0.5-abs(U); // >=0, <=0.5
		if(us==0 || (us<0.013 && V>us)) { // add test to deal with us==0 case
			continue;
		}
		// (2.0*a/us+b)  >=9.58267
		// (2.0*a/us+b)*U  anything
		double k=std::floor((2.0*a/us+b)*U+u+0.445); // anything
		double inv_alpha=1.1239+1.1328/(b-3.4); // >=1.1239, <1.3287
		V=V*inv_alpha/(a/(us*us)+b);
		if(k>=10.0) {
			if(std::log(V*smu)<=(k+0.5)*log(u/k)-u-log(sqrt(2*M_PI))+k-(1.0/12.0-1.0/(360*k*k))/k) {
				return k;
			}
		} else if(0<=k && std::log(V)<k*std::log(u)-u-log_fact_table[k]) {
			return k;
		}
	}
}
