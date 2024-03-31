#include <cstdint>
#include <array>
#if __x86_64 || _M_X64
#include <emmintrin.h>
#include <smmintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#elif __aarch64__
#include <arm_neon.h>
#endif

const uint64_t P_LN2_INV_2_POW_63=13306513097844322492ULL;

#ifdef _MSC_VER // windows

#define clz64 (uint32_t)__lzcnt64
#define clz32 (uint32_t)__lzcnt
#define popcount __popcnt

#elif defined(__GNUC__) || defined(__clang__) // gcc/clang

#define clz64 __builtin_clzll
#define clz32 __builtin_clz
#define popcount __builtin_popcount

#endif

constexpr std::array<int64_t,10> log_fact_table_fixed = {{
	          0ULL,
	          0ULL,
	 2977044472ULL,
	 7695548323ULL,
	13649637266ULL,
	20562120465ULL,
	28257668788ULL,
	36615289239ULL,
	45546422654ULL,
	54983430356ULL
}};

#ifdef _MSC_VER // Windows

static inline uint64_t multu64hi(uint64_t x, uint64_t y) {
	unsigned __int64 ret;
	_umul128(x, y, &ret);
	return ret;
}
static inline void multu64hilo(uint64_t x, uint64_t y, uint64_t* rhi, uint64_t* rlo) {
	*rlo=_umul128(x, y, rhi);
}

// TODO Windows version of fixed_mult64s and fixed_mult64u coming soon.

#elif defined(__GNUC__) || defined(__clang__) // gcc/clang

static inline uint64_t multu64hi(uint64_t x,uint64_t y) {
	return (uint64_t)((((unsigned __int128)x)*y)>>64);
}

static inline int64_t mults64hi(int64_t x,int64_t y) {
	return (int64_t)((((__int128)x)*y)>>64);
}

static inline void multu64hilo(uint64_t x,uint64_t y,uint64_t* rhi,uint64_t* rlo) {
	unsigned __int128 ret=((unsigned __int128)x)*y;
	*rhi=ret>>64;
	*rlo=ret;
}

static inline int64_t fixed_mult64s(int64_t x,int64_t y) {
	return (int64_t)((((__int128)x)*y)>>32);
}

static inline int64_t fixed_mult64u(uint64_t x,uint64_t y) {
	return (uint64_t)((((unsigned __int128)x)*y)>>32);
}

#endif

static inline uint32_t multu32hi(uint32_t x,uint32_t y) {
	return (((uint64_t)x)*y)>>32;
}

static inline uint32_t mults32hi(int32_t x,int32_t y) {
	return (((int64_t)x)*y)>>32;
}

static inline uint32_t p_exp2_32_internal(uint32_t x) {
	uint32_t u=58831021U;
	u=multu32hi(u,x)+222008398U;
	u=multu32hi(u,x)+1037829222U;
	u=multu32hi(u,x)+(2976266834U+0x7C4F);
	return (multu32hi(u,x)>>2)+0x40000000U;
}

int64_t log_64_fixed(uint64_t lx) {
	int32_t lead=clz64(lx);
	int32_t x=((lx<<lead)>>32)-0x80000000ULL;
	// x is a 1.63 unsigned fixed in the range [0,1)
	// calculate ln2(x+1)
	// result is 1.63 unsigned fixed in the range [0,1)
	int32_t u=          -19518282;
	u=mults32hi(u<<1,x) +109810370;
	u=mults32hi(u<<1,x) -291900857;
	u=mults32hi(u<<1,x) +516277066;
	u=mults32hi(u<<1,x) -744207376;
	u=mults32hi(u<<1,x) +1027494097;
	u=mults32hi(u<<1,x) -1548619616;
	u=mults32hi(u,x)+   (1549074032+93);
	uint64_t d=mults32hi(u,x)<<3;
	return mults64hi((d+((31LL-lead)<<32)),6393154322601327829LL)<<1;
}

uint64_t fixed_sqrt_32_32(uint64_t x) {
	uint32_t lead=clz64(x)>>1;
	x<<=(lead<<1);
	if((1ULL<<62)==x) {
		return x>>(lead+15);
	}
	// x is a 2.62 unsigned fixed in the range (1,4)
	// result is 1.63 unsigned fixed in the range (0.5,1)
	// start with a linear interpolation correct at the endpoints
	// 7/6 - 1/6 x, so 1->1, 4->0.5
	uint64_t y=3074457345618258602ULL-multu64hi(x,12297829382473034410ULL);
	// now do some Newton-Raphson
	// y=y*(3/2-1/2*x*y*y)
	// Maximum error for #iterations:
	// 0	~0.2
	// 1	~0.06
	// 2	~0.005
	// 3	~3e-5
	// 5	~1e-9 (about 30 bits - limit of the algorithm)
	y=multu64hi(y,0xC000000000000000ULL-((multu64hi(multu64hi(y,y),x))))<<1;
	y=multu64hi(y,0xC000000000000000ULL-((multu64hi(multu64hi(y,y),x))))<<1;
	y=multu64hi(y,0xC000000000000000ULL-((multu64hi(multu64hi(y,y),x))))<<1;
	y=multu64hi(y,0xC000000000000000ULL-((multu64hi(multu64hi(y,y),x)))); // dont shift left on the last one
	return multu64hi(y,x)>>(lead+14);
}

union variant16 {
#if __x86_64
	__m128i v;
#elif __aarch64__
	uint16x8_t v;
#endif
	uint16_t s16[8];
	uint64_t s64[2];
};

#if __x86_64
typedef __m128i uint16x8_t;
#elif __aarch64__
#else
typedef variant16 uint16x8_t;
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

static inline uint64_t horizonal_mult8_16_corr(uint16x8_t x) {
#if __x86_64 || _M_X64
	union variant16 u;
	u.v=x;
	uint64_t startx0=u.s16[0];
	uint64_t startx1=u.s16[4];
	for(uint32_t i=1;i<4;i++) {
		startx0*=u.s16[i];
		startx1*=u.s16[i+4];
	}
#elif __aarch64__
	uint32x4_t t1=vmull_u16(vget_low_u16(x),vget_high_u16(x));
	uint64x2_t t2=vmull_u32(vget_low_u32(t1),vget_high_u32(t1));
	uint64_t startx0=vdupd_laneq_u64(t2,0);
	uint64_t startx1=vdupd_laneq_u64(t2,1);
#else
	uint64_t startx0=x.s16[0];
	uint64_t startx1=x.s16[4];
	for(uint32_t i=1;i<4;i++) {
		startx0*=x.s16[i];
		startx1*=x.s16[i+4];
	}
#endif
	return multu64hi(startx0,startx1);
}

uint32_t poisson_random_variable_fixed_int(uint64_t* seed, int64_t lambda) {
	if(lambda<=0) {
		return 0;
	}
	if(lambda<=77309411328LL) { // 18
		uint64_t num_digits=multu64hi(lambda,P_LN2_INV_2_POW_63)<<1;
		int32_t int_digits=num_digits>>32;
		uint64_t start=((uint64_t)p_exp2_32_internal(num_digits&0xFFFFFFFF))<<33;
		uint32_t ret=-1;
		while(int_digits>=0) {
			uint64_t x=(fast_rand64(seed)|1);
			start=multu64hi(start,x);
			uint32_t z=clz64(start);
			int_digits-=z;
			start<<=z;
			ret++;
		}
		return ret;
	}
	if(lambda<=163208757248LL) { // 38
		uint64_t num_digits=multu64hi(lambda,P_LN2_INV_2_POW_63)<<1;
		int32_t int_digits=num_digits>>32;
		num_digits&=0xFFFFFFFFULL;
		uint32_t r7=p_exp2_32_internal((uint32_t)num_digits)>>15; // e^(ln(2)*x) = (e^ln(2))^x = 2^x
		int32_t ret=-1;
		uint16_t old_start_flag=r7;
		int32_t old_int_digits=int_digits;
#if __x86_64 || _M_X64
		__m128i zero=_mm_setzero_si128();
		__m128i const_1=_mm_set1_epi16(1);
		__m128i const_FF00=_mm_set1_epi16(0xFF00U);
		__m128i const_F000=_mm_set1_epi16(0xF000U);
		__m128i const_C000=_mm_set1_epi16(0xC000U);
		__m128i const_8000=_mm_set1_epi16(0x8000U);
		__m128i old_start=_mm_insert_epi16(_mm_set1_epi16(0xFFFFU),r7,0);
		uint64_t a=fast_rand64(seed);
		uint64_t b=fast_rand64(seed);
		__m128i old_rand=_mm_set_epi64x(b,a);
		__m128i startx=_mm_insert_epi16(old_rand,(((uint32_t)_mm_extract_epi16(old_rand,0))*r7)>>16,0);
		startx=_mm_or_si128(startx,const_1);
		__m128i clz_select8=_mm_cmpeq_epi16(_mm_and_si128(startx,const_FF00),zero);
		startx=_mm_blendv_epi8(startx,_mm_slli_epi16(startx,8),clz_select8);
		__m128i clz_select4=_mm_cmpeq_epi16(_mm_and_si128(startx,const_F000),zero);
		startx=_mm_blendv_epi8(startx,_mm_slli_epi16(startx,4),clz_select4);
		__m128i clz_select2=_mm_cmpeq_epi16(_mm_and_si128(startx,const_C000),zero);
		startx=_mm_blendv_epi8(startx,_mm_slli_epi16(startx,2),clz_select2);
		__m128i clz_select1=_mm_cmpeq_epi16(_mm_and_si128(startx,const_8000),zero);
		startx=_mm_blendv_epi8(startx,_mm_slli_epi16(startx,1),clz_select1);
		uint32_t t=__builtin_popcount(_mm_movemask_epi8(clz_select8));
		t=__builtin_popcount(_mm_movemask_epi8(clz_select4))+t+t;
		t=__builtin_popcount(_mm_movemask_epi8(clz_select2))+t+t;
		t=__builtin_popcount(_mm_movemask_epi8(clz_select1))+t+t;
		int_digits-=(t>>1);
#elif __aarch64__
		uint16x8_t const_1=vdupq_n_u16(1);
		uint16x8_t old_start=vsetq_lane_u16(r7,vdupq_n_u16(0xFFFF),0);
		uint64_t a=fast_rand64(seed);
		uint64_t b=fast_rand64(seed);
		uint16x8_t old_rand=vcombine_u16(vcreate_u16(a),vcreate_u16(b));
		uint16x8_t startx=vsetq_lane_u16((((uint32_t)vgetq_lane_u16(old_rand,0))*r7)>>16,old_rand,0);
		startx=vorrq_u16(startx,const_1);
		uint16x8_t zz=vclzq_u16(startx);
		startx=vshlq_u16(startx,vreinterpretq_s16_u16(zz));
		int_digits-=vaddvq_u16(zz);
#else // don't know what the processor is
		variant16 startx,old_start,old_rand;
		old_rand.s64[0]=(fast_rand64(seed));
		old_rand.s64[1]=(fast_rand64(seed));
		startx=old_rand;
		startx.s16[0]=((((uint32_t)old_rand.s16[0])*r7)>>16)|1U;
		for(uint32_t i=0;i<8;i++) {
			old_start.s16[i]=0xFFFF;
			uint16_t x=startx.s16[i]|(uint16_t)1;
			int32_t z=clz32(x)-16;
			int_digits-=z;
			x<<=z;
			startx.s16[i]=x;
		}
		old_start.s16[0]=r7;
#endif
		ret += 8;
		uint16x8_t old_old_start=old_start;
		uint16_t old_old_start_flag=old_start_flag;
		int32_t old_old_int_digits=old_int_digits;
		uint16x8_t old_old_rand=old_rand;
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
			startx=_mm_mulhi_epu16(startx,old_rand);
			startx=_mm_or_si128(startx,const_1);
			__m128i clz_select8=_mm_cmpeq_epi16(_mm_and_si128(startx,const_FF00),zero);
			startx=_mm_blendv_epi8(startx,_mm_slli_epi16(startx,8),clz_select8);
			__m128i clz_select4=_mm_cmpeq_epi16(_mm_and_si128(startx,const_F000),zero);
			startx=_mm_blendv_epi8(startx,_mm_slli_epi16(startx,4),clz_select4);
			__m128i clz_select2=_mm_cmpeq_epi16(_mm_and_si128(startx,const_C000),zero);
			startx=_mm_blendv_epi8(startx,_mm_slli_epi16(startx,2),clz_select2);
			__m128i clz_select1=_mm_cmpeq_epi16(_mm_and_si128(startx,const_8000),zero);
			startx=_mm_blendv_epi8(startx,_mm_slli_epi16(startx,1),clz_select1);
			uint32_t t=__builtin_popcount(_mm_movemask_epi8(clz_select8));
			t=__builtin_popcount(_mm_movemask_epi8(clz_select4))+t+t;
			t=(__builtin_popcount(_mm_movemask_epi8(clz_select1))>>1)+__builtin_popcount(_mm_movemask_epi8(clz_select2))+t+t;
			int_digits-=t;
#elif __aarch64__
			uint64_t a=fast_rand64(seed);
			uint64_t b=fast_rand64(seed);
			old_rand=vcombine_u16(vcreate_u16(a),vcreate_u16(b));
			uint32x4_t mul_lo=vmull_u16(vget_low_u16(startx),vget_low_u16(old_rand));
			uint32x4_t mul_hi=vmull_high_u16(startx,old_rand);
			startx=vcombine_u16(vshrn_n_u32(mul_lo,16),vshrn_n_u32(mul_hi,16));
			uint16x8_t mult=vorrq_u16(startx,const_1);
			uint16x8_t z=vclzq_u16(mult);
			startx=vshlq_u16(mult,vreinterpretq_s16_u16(z));
			int_digits-=vaddvq_u16(z);
#else // don't know what the processor is
#warning noopt
			old_rand.s64[0]=(fast_rand64(seed));
			old_rand.s64[1]=(fast_rand64(seed));
			for(uint32_t i=0;i<8;i++) {
				uint32_t x=startx.s16[i];
				x=((x*old_rand.s16[i])>>16)|1U;
				int32_t z=clz32(x)-16;
				int_digits-=z;
				x<<=z;
				startx.s16[i]=x;
			}
#endif
			ret+=8;
		}
		union variant16 urand;
		ret-=8;
		uint64_t start64=horizonal_mult8_16_corr(old_start);
		int32_t z=clz64(start64);
		if(old_start_flag==0 && old_int_digits<z) {
			ret-=8;
			int_digits=old_old_int_digits;
#if __x86_64 || _M_X64 || __aarch64__
			urand.v=old_old_rand;
#else
			urand=old_old_rand;
#endif
			old_start_flag=old_old_start_flag;
			start64=horizonal_mult8_16_corr(old_old_start);
			z=clz64(start64);
		} else {
			int_digits=old_int_digits;
#if __x86_64 || _M_X64 || __aarch64__
			urand.v=old_rand;
#else
			urand=old_rand;
#endif
		}
		uint16_t start;
		if(old_start_flag==0) {
			int_digits-=z;
			start=(uint16_t)(start64>>(48-z));
		} else {
			start=old_start_flag;
		}
		uint32_t i=0;
		while(int_digits>=0 && i<8) {
			start=((((uint32_t)start)*urand.s16[i++])>>16)|1U;
			int32_t z=clz32(start)-16;
			int_digits-=z;
			start<<=z;
			ret++;
		}
		return ret;
	}
	uint64_t iu=lambda;
	while(true) {
		//double smu=std::sqrt(u); // >=3.1623 <=10000
		uint64_t ismu=fixed_sqrt_32_32(iu);
		//double b=0.931+2.53*smu; // >=8.9316 <=25300
		uint64_t ib=3998614553ULL+(multu64hi(ismu,11667565626621291397ULL)<<2);
		//double a=-0.059+0.02483*b; // >=0.16277 <=629
		uint64_t ia=multu64hi(ib,458032655350208166ULL)-253403070ULL;
		//double vr=0.9277-3.6224/(b-2.0); // >=0.4051, <=0.9277
		uint64_t ivr=3984441160ULL-((16705371433151369943ULL/(ib-8589934592ULL))<<2);
		uint64_t iV=fast_rand64(seed)>>32;
		//if(V<0.86*vr) { // V/vr<0.86
		if(iV<multu64hi(15864199903390214389ULL,ivr)) {
			//double U=V/vr-0.43; // >=-0.43, <=0.43
			int64_t iU=(iV<<32)/ivr-1846835937ULL;
			//double us=0.5-abs(U); // >=0.07, <=0.5
			uint64_t ius=2147483648ULL-abs(iU);
			//uint64_t k=std::floor((2.0*a/us+b)*U+u+0.445);
			uint64_t i2a_div_us=((ia<<21)/ius)<<12; //udiv64fixed(ia<<1,ius);
			uint64_t ik=(fixed_mult64s(i2a_div_us+ib,iU)+iu+1911260447ULL)>>32;
			return ik;
		}
		uint64_t it=fast_rand64(seed)>>32;
		//double t=it/4294967296.0;
		int64_t iU;
		if(iV>=ivr) {
			//U=t-0.5; // >=-0.5, <=0.5
			iU=it-2147483648ULL;
		} else {
			//U=V/vr-0.93; // >=-0.93, <=0.07
			iU=(iV<<32)/ivr-3994319585ULL;
			//U=((U<0)?-0.5:0.5)-U; // >=-0.5, <=0.5
			iU=((iU<0)?-2147483648LL:2147483648LL)-iU;
			//V=t*vr; // >=0, <=0.9277
			iV=fixed_mult64u(it,ivr);
		}
		//double us=0.5-abs(U); // >=0, <=0.5
		uint64_t ius=2147483648ULL-abs(iU);
		if(ius<65536 || (ius<55834575ULL && iV>ius)) {
			continue;
		}
		//double k=std::floor((2.0*a/us+b)*U+u+0.445); // anything
		uint64_t i2a_div_us=((ia<<21)/ius)<<12; //udiv64fixed(ia<<1,ius);
		int64_t ik=(fixed_mult64s(i2a_div_us+ib,iU)+(int64_t)iu+1911260447LL)>>32;
		//double inv_alpha=1.1239+1.1328/(b-3.4); // >=1.1239, <1.3287
		uint64_t iinv_alpha=4827113744ULL+((10448235843349090035ULL/(ib-14602888806ULL))<<1);
		//V=V*inv_alpha/(a/(us*us)+b);
		iV=((fixed_mult64u(iinv_alpha,iV)<<31)/((((ia<<20)/fixed_mult64u(ius,ius))<<12)+ib)<<1);
		if(ik>=10.0) {
			int64_t lhs=log_64_fixed(fixed_mult64u(iV,ismu));
			int64_t rhs=((ik<<1)+1)*((log_64_fixed(iu/ik))>>1)-iu-3946810947LL+(ik<<32);
			if(lhs>rhs) {
				continue;
			}
			rhs-=(357913941ULL-(4294967296ULL/(360*ik*ik)))/ik;
			if(lhs>rhs) {
				continue;
			}
			return ik;
		} else if(0<=ik && log_64_fixed(iV)<((int64_t)ik)*log_64_fixed(iu)-(int64_t)iu-log_fact_table_fixed[ik]) {
			return ik;
		}
	}
}
