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
			uint32_t p=poisson_random_variable_fixed_int(&seed,lambda*4294967296ULL);
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
