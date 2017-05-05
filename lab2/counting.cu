#include "counting.h"
#include <cstdio>
#include <cassert>
#include <stdlib.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }
struct is_space
{
  __host__ __device__
  bool operator()(char x)
  {
    return x <= ' '; 
  }
};


struct zero
{
  __host__ __device__
  bool operator()(int x)
  {
    return 0;
  }
};

void CountPosition1(const char *text, int *pos, int text_size)
{
	struct is_space is_space;
	struct zero zero;
	thrust::fill(thrust::device, pos, pos + text_size, 1);
	thrust::transform_if(thrust::device, text, text + text_size, pos, zero, is_space);
	thrust::inclusive_scan_by_key(thrust::device, pos, pos + text_size, pos, pos);
}
__global__ void labeling(const char *text, int *pos, int text_size){
	int index = threadIdx.x*blockDim.y+threadIdx.y + blockDim.x*blockDim.y*(gridDim.y*blockIdx.x + blockIdx.y);
	if (index >= text_size) {
		return;
	}
	pos[index] = 0;
	if (text[index] <= ' ')
		return ;
	for (int k = index; k >= 0; k--) {
		if (text[k] <= ' ') {
			pos[index] = index - k;
			return;
		}
	}
	pos[index] = index+1;

}
void CountPosition2(const char *text, int *pos, int text_size)
{
	labeling<<<dim3(1024,1024), dim3(8, 8)>>>(text, pos, text_size);
}
