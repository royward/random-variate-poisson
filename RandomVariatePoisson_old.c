// BSD 3-Clause License
// 
// Copyright (c) 2023, Roy Ward
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "RandomVariatePoisson.h"
#if __x86_64 || _M_X64
#include <emmintrin.h>
#include <smmintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#elif __aarch64__
#include <arm_neon.h>
#endif

#ifdef _MSC_VER // windows
#define clz64 (uint32_t)__lzcnt64
#define clz32 (uint32_t)__lzcnt
#define __builtin_popcount __popcnt
static inline uint64_t multu64hi(uint64_t x, uint64_t y) {
	unsigned __int64 ret;
	_umul128(x, y, &ret);
	return ret;
}
static inline void multu64hilo(uint64_t x, uint64_t y, uint64_t* rhi, uint64_t* rlo) {
	*rlo=_umul128(x, y, rhi);
}
#elif defined(__GNUC__) || defined(__clang__) // gcc/clang
#define clz64 __builtin_clzll
#define clz32 __builtin_clz
static inline uint64_t multu64hi(uint64_t x,uint64_t y) {
	return (uint64_t)((((unsigned __int128)x)*y)>>64);
}
static inline void multu64hilo(uint64_t x,uint64_t y,uint64_t* rhi,uint64_t* rlo) {
	unsigned __int128 ret=((unsigned __int128)x)*y;
	*rhi=ret>>64;
	*rlo=ret;
}
#else
#error Compiler is not gcc, clang or Visual Studio. Need to define clz and 128 bit arithmetic for your compiler
#endif

static inline uint32_t max(uint32_t x, uint32_t y) {
	return (x<y)?y:x;
}

static inline uint64_t fast_rand64(uint64_t* seed) {
	*seed += 0x60bee2bee120fc15ULL;
	uint64_t hi,lo;
	multu64hilo(*seed,0xa3b195354a39b70dULL,&hi,&lo);
	uint64_t m1 = hi^lo;
	multu64hilo(m1,0x1b03738712fad5c9ULL,&hi,&lo);
	uint64_t m2 = hi^lo;
	return m2;
}

const uint64_t P_LN2_INV_2_POW_63=13306513097844300000ULL;

// ./lolremez --long-double -d 4 -r "0:1" "exp2(x)" --debug
// [debug] error: 3.7044659369797384313e-6
// Approximation of f(x) = exp2(x)
// on interval [ 0, 1 ]
// with a polynomial of degree 4.
// long double f(long double x) {
//     long double u = 1.3697664475809267368e-2l;
//     u = u * x + 5.1690358205939469451e-2l;
//     u = u * x + 2.4163844572498163183e-1l;
//     u = u * x + 6.9296612266139567188e-1l;
//     return u * x + 1.0000037044659369797l;
// }

static inline uint32_t multu32hi(uint32_t x,uint32_t y) {
	return (((uint64_t)x)*y)>>32;
}

static inline uint32_t p_exp2_32_internal(uint32_t x) {
	uint32_t u=58831021U;
	u=multu32hi(u,x)+222008398U;
	u=multu32hi(u,x)+1037829222U;
	u=multu32hi(u,x)+(2976266834U+0x7C4F);
	return (multu32hi(u,x)>>2)+0x40000000U;
}

union variant {
#if __x86_64 || _M_X64
	__m128i v;
#elif __aarch64__
	uint8x16_t v;
#endif
	uint8_t s8[16];
	uint64_t s64[2];
};

#if __x86_64 || _M_X64
typedef __m128i uint8x16_t;
#elif __aarch64__
#else
typedef variant uint8x16_t;
#endif

static inline uint64_t horizonal_mult16_8_corr(uint8x16_t x) {
#if __x86_64 || _M_X64
	union variant u;
	u.v=x;
	uint64_t startx0=u.s8[0];
	uint64_t startx1=u.s8[8];
	for(uint32_t i=1;i<8;i++) {
		startx0*=u.s8[i];
		startx1*=u.s8[i+8];
	}
#elif __aarch64__
	uint16x8_t t0=vmull_u8(vget_low_u8(x),vget_high_u8(x));
	uint32x4_t t1=vmull_u16(vget_low_u16(t0),vget_high_u16(t0));
	uint64x2_t t2=vmull_u32(vget_low_u32(t1),vget_high_u32(t1));
	uint64_t startx0=vdupd_laneq_u64(t2,0);
	uint64_t startx1=vdupd_laneq_u64(t2,1);
#else
	uint64_t startx0=x.s8[0],startx1=x.s8[8];
	for(uint32_t i=1;i<8;i++) {
		startx0*=x.s8[i];
		startx1*=x.s8[i+8];
	}
#endif
	return (multu64hi(startx0,startx1)>>8)*(256+36);
}

// lambda is fixed 32.32
uint32_t poisson_random_variable_fixed_int(uint64_t* seed, int64_t lambda) {
	if(lambda<=0) {
		return 0;
	}
	lambda=multu64hi(9224486805748300000ULL,(lambda<<1));
	uint64_t num_digits=multu64hi(lambda,P_LN2_INV_2_POW_63)<<1;
	int32_t int_digits=num_digits>>32;
	num_digits&=0xFFFFFFFFULL;
	uint32_t r15=p_exp2_32_internal((uint32_t)num_digits)>>23; // e^(ln(2)*x) = (e^ln(2))^x = 2^x
	int32_t ret=-1;
	if (int_digits<18) {
		uint8_t start=r15;
		uint32_t i=8;
		uint64_t r=0;
		while(int_digits>=0) {
			if(i==8) {
				r=fast_rand64(seed);
				i=0;
			}
			start=max((((uint32_t)start)*(((uint32_t)r)&0xFF)+0xA7)>>8,1U);
			uint32_t z=clz32(start)-24;
			int_digits-=z;
			start<<=z;
			ret++;
			i++;
			r>>=8;
		}
		return ret;
	}
	uint8_t old_start_flag=r15;
	int32_t old_int_digits=int_digits;
#if __x86_64 || _M_X64
	__m128i zero=_mm_setzero_si128();
	__m128i const_1=_mm_set1_epi8(1);
	__m128i const_F0=_mm_set1_epi8(0xF0U);
	__m128i const_C0=_mm_set1_epi8(0xC0U);
	__m128i const_80=_mm_set1_epi8(0x80U);
	__m128i const_FC=_mm_set1_epi8(0xFCU);
	__m128i const_FE=_mm_set1_epi8(0xFEU);
	__m128i const_A7=_mm_set1_epi16(0xA7U);
	__m128i old_start=_mm_insert_epi8(_mm_set1_epi8(0xFFU),r15,0);
	uint64_t a=fast_rand64(seed);
	uint64_t b=fast_rand64(seed);
	__m128i old_rand=_mm_set_epi64x(b,a);
	__m128i startx=_mm_insert_epi8(old_rand,(((uint32_t)((uint8_t)_mm_extract_epi8(old_rand,0)))*r15+0xA7)>>8,0);
	startx=_mm_max_epu8(startx,const_1);
	__m128i clz_select4=_mm_cmpeq_epi8(_mm_and_si128(startx,const_F0),zero);
	startx=_mm_blendv_epi8(startx,_mm_and_si128(_mm_slli_epi64(startx,4),const_F0),clz_select4);
	__m128i clz_select2=_mm_cmpeq_epi8(_mm_and_si128(startx,const_C0),zero);
	startx=_mm_blendv_epi8(startx,_mm_and_si128(_mm_slli_epi64(startx,2),const_FC),clz_select2);
	__m128i clz_select1=_mm_cmpeq_epi8(_mm_and_si128(startx,const_80),zero);
	startx=_mm_blendv_epi8(startx,_mm_and_si128(_mm_slli_epi64(startx,1),const_FE),clz_select1);
	uint32_t t=__builtin_popcount(_mm_movemask_epi8(clz_select4));
	t=__builtin_popcount(_mm_movemask_epi8(clz_select2))+t+t;
	t=__builtin_popcount(_mm_movemask_epi8(clz_select1))+t+t;
	int_digits-=t;
#elif __aarch64__
	uint16x8_t const_a7=vdupq_n_u16(0xA7);
	uint8x16_t const_1=vdupq_n_u8(1);
	uint8x16_t old_start=vsetq_lane_u8(r15,vdupq_n_u8(0xFF),0);
	uint64_t a=fast_rand64(seed);
	uint64_t b=fast_rand64(seed);
	uint8x16_t old_rand=vcombine_u8(vcreate_u8(a),vcreate_u8(b));
	uint8x16_t startx=vsetq_lane_u8((((uint32_t)((uint8_t)vgetq_lane_u8(old_rand,0)))*r15+0xA7)>>8,old_rand,0);
	startx=vmaxq_u8(startx,const_1);
	uint8x16_t zz=vclzq_u8(startx);
	startx=vshlq_u8(startx,vreinterpretq_s8_u8(zz));
	int_digits-=vaddvq_u8(zz);
#else // don't know what the processor is
	variant startx,old_start,old_rand;
	old_rand.s64[0]=(fast_rand64(seed));
	old_rand.s64[1]=(fast_rand64(seed));
	startx=old_rand;
	startx.s8[15]=max((((uint32_t)old_rand.s8[15])*r15+0xA7)>>8,1U);
	for(uint32_t i=0;i<16;i++) {
		old_start.s8[i]=0xFF;
		uint8_t x=max(startx.s8[i],(uint8_t)1);
		int32_t z=clz32(x)-24;
		int_digits-=z;
		x<<=z;
		startx.s8[i]=x;
	}
	old_start.s8[15]=r15;
#endif
	ret += 16;
	uint8x16_t old_old_start=old_start;
	uint8_t old_old_start_flag=old_start_flag;
	int32_t old_old_int_digits=old_int_digits;
	uint8x16_t old_old_rand=old_rand;
	while (int_digits >= 0) {
		old_old_start=old_start;
		old_old_start_flag=old_start_flag;
		old_old_int_digits=old_int_digits;
		old_old_rand=old_rand;
		old_start=startx;
		old_start_flag=0;
		old_int_digits=int_digits;
#if __x86_64 || _M_X64
		uint64_t a=fast_rand64(seed);
		uint64_t b=fast_rand64(seed);
		old_rand=_mm_set_epi64x(b,a);
		__m128i start_lo=_mm_unpacklo_epi8(startx,zero);
		__m128i start_hi=_mm_unpackhi_epi8(startx,zero);
		__m128i x_lo=_mm_unpacklo_epi8(old_rand,zero);
		__m128i x_hi=_mm_unpackhi_epi8(old_rand,zero);
		start_lo=_mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(start_lo,x_lo),const_A7),8);
		start_hi=_mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(start_hi,x_hi),const_A7),8);
		startx=_mm_packus_epi16(start_lo,start_hi);
		startx=_mm_max_epu8(startx,const_1);
		__m128i clz_select4=_mm_cmpeq_epi8(_mm_and_si128(startx,const_F0),zero);
		startx=_mm_blendv_epi8(startx,_mm_and_si128(_mm_slli_epi64(startx,4),const_F0),clz_select4);
		__m128i clz_select2=_mm_cmpeq_epi8(_mm_and_si128(startx,const_C0),zero);
		startx=_mm_blendv_epi8(startx,_mm_and_si128(_mm_slli_epi64(startx,2),const_FC),clz_select2);
		__m128i clz_select1=_mm_cmpeq_epi8(_mm_and_si128(startx,const_80),zero);
		startx=_mm_blendv_epi8(startx,_mm_and_si128(_mm_slli_epi64(startx,1),const_FE),clz_select1);
		uint32_t t=__builtin_popcount(_mm_movemask_epi8(clz_select4));
		t=__builtin_popcount(_mm_movemask_epi8(clz_select2))+t+t;
		t=__builtin_popcount(_mm_movemask_epi8(clz_select1))+t+t;
		int_digits-=t;
#elif __aarch64__
		uint64_t a=fast_rand64(seed);
		uint64_t b=fast_rand64(seed);
		old_rand=vcombine_u8(vcreate_u8(a),vcreate_u8(b));
		uint16x8_t mul_lo=vmull_u8(vget_low_u8(startx),vget_low_u8(old_rand));
		uint16x8_t mul_hi=vmull_high_u8(startx,old_rand);
		startx=vcombine_u8(vshrn_n_u16(vaddq_u16(mul_lo,const_a7),8),vshrn_n_u16(vaddq_u16(mul_hi,const_a7),8));
		uint8x16_t mult=vmaxq_u8(startx,const_1);
		uint8x16_t z=vclzq_u8(mult);
		startx=vshlq_u8(mult,vreinterpretq_s8_u8(z));
		int_digits-=vaddvq_u8(z);
#else // don't know what the processor is
		old_rand.s64[0]=(fast_rand64(seed));
		old_rand.s64[1]=(fast_rand64(seed));
		for(uint32_t i=0;i<16;i++) {
			uint32_t x=startx.s8[i];
			x=max((x*old_rand.s8[i]+0xA7)>>8,1U);
			int32_t z=clz32(x)-24;
			int_digits-=z;
			x<<=z;
			startx.s8[i]=x;
		}
#endif
		ret+=16;
	}
	union variant urand;
	ret-=16;
	uint64_t start64=horizonal_mult16_8_corr(old_start);
	int32_t z=clz64(start64);
	if(old_start_flag==0 && old_int_digits<z) {
		ret-=16;
		int_digits=old_old_int_digits;
#if __x86_64 || _M_X64 || __aarch64__
		urand.v=old_old_rand;
#else
		urand=old_old_rand;
#endif
		old_start_flag=old_old_start_flag;
		start64=horizonal_mult16_8_corr(old_old_start);
		z=clz64(start64);
	} else {
		int_digits=old_int_digits;
#if __x86_64 || _M_X64 || __aarch64__
		urand.v=old_rand;
#else
		urand=old_rand;
#endif
	}
	uint8_t start;
	if(old_start_flag==0) {
		int_digits-=z;
		start=(uint8_t)(start64>>(56-z));
	} else {
		start=old_start_flag;
	}
	uint32_t i=0;
	while(int_digits>=0 && i<16) {
		start=max((((uint32_t)start)*urand.s8[i++]+0xA7)>>8,1U);
		int32_t z=clz32(start)-24;
		int_digits-=z;
		start<<=z;
		ret++;
	}
	return ret;
}
