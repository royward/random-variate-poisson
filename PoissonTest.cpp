#include "RandomVariatePoisson.h"
#include <iostream>
#include <iomanip>
#include <cstring>

using namespace std;

static const uint32_t ITERATIONS=10000;
static const uint32_t MAX=10000;

int main() {
	uint64_t seed=1234123452347;
	uint32_t* hist1=new uint32_t[MAX];
	for(uint64_t lambda=0;lambda<100;lambda+=1) {
	//for(double lambda=0;lambda<99999;lambda+=1000) {
		uint64_t tot1=0;
		memset(hist1,0,MAX*4);
		for(uint32_t i=0;i<ITERATIONS;i++) {
			uint32_t p=poisson_random_variable_fixed_int(seed,lambda*4294967296ULL);
			if(p>=MAX) {
				p=MAX-1;
			}
			hist1[p]++;
		}
		//uint64_t end_time=chrono::system_clock::now().time_since_epoch().count();
		for(uint32_t i=0;i<MAX;i++) {
 			tot1+=hist1[i]*i;
		}
		cout << lambda << ',' << setprecision(15) << (1.0*tot1)/ITERATIONS-lambda;
		cout << endl;
	}
	return 0;
}
