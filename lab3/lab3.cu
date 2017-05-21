#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}
__device__ void assign_add(float *target, const float *source)
{
	target[0] += source[0];
	target[1] += source[1];
	target[2] += source[2];
}
__device__ void assign_sub(float *target, const float *source)
{
	target[0] -= source[0];
	target[1] -= source[1];
	target[2] -= source[2];
}
__device__ void fill_boundry(float *sum, const int yt, const int xt, const int ht, const int wt,
				const float *background, const int wb, const int hb, const int oy, const int ox)
{	
	float fill[3] = {255.0f, 255.0f, 255.0f};
	int yb, xb, curb;
	if(yt-1 == -1){
		yb = oy+yt-1, xb = ox+xt;
		yb = (yb < hb) ? yb : hb-1;
		xb = (xb < wb) ? xb : wb-1;
		yb = (yb < 0) ? 0 : yb;
		xb = (xb < 0) ? 0 : xb;
		curb = wb*yb+xb;
		assign_add(sum, &background[curb*3]);
		assign_sub(sum, fill);
	}
	if(xt-1 == -1){
		yb = oy+yt, xb = ox+xt-1;
		yb = (yb < hb) ? yb : hb-1;
		xb = (xb < wb) ? xb : wb-1;
		yb = (yb < 0) ? 0 : yb;
		xb = (xb < 0) ? 0 : xb;
		curb = wb*yb+xb;
		assign_add(sum, &background[curb*3]);
		assign_sub(sum, fill);
	}
	if(yt+1 == ht){
		yb = oy+yt+1, xb = ox+xt;
		yb = (yb < hb) ? yb : hb-1;
		xb = (xb < wb) ? xb : wb-1;
		yb = (yb < 0) ? 0 : yb;
		xb = (xb < 0) ? 0 : xb;
		curb = wb*yb+xb;
		assign_add(sum, &background[curb*3]);
		assign_sub(sum, fill);
	}
	if(xt+1 == wt){
		yb = oy+yt, xb = ox+xt+1;
		yb = (yb < hb) ? yb : hb-1;
		xb = (xb < wb) ? xb : wb-1;
		yb = (yb < 0) ? 0 : yb;
		xb = (xb < 0) ? 0 : xb;
		curb = wb*yb+xb;
		assign_add(sum, &background[curb*3]);
		assign_sub(sum, fill);
	}
}
__global__ void	CalculateFixed(
	const float *background, const float *target, const float *mask, float *fixed,
	const int wb, const int hb, const int wt, const int ht, const int oy, const int ox
	)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	const int Nt = wt*(yt-1)+xt;
	const int Wt = wt*yt+xt-1;
	const int St = wt*(yt+1)+xt;
	const int Et = wt*yt+xt+1;
	float sum[3] = {};
	if(yt < ht and xt < wt){
		int yb, xb, curb;
		if((yt-1) >= 0){	
			assign_sub(sum, &target[Nt*3]);
			if(mask[Nt] < 127.0f){
				yb = oy+yt-1, xb = ox+xt;
				yb = (yb < hb) ? yb : hb-1;
				xb = (xb < wb) ? xb : wb-1;
				yb = (yb < 0) ? 0 : yb;
				xb = (xb < 0) ? 0 : xb;
				curb = wb*yb+xb;
				assign_add(sum, &background[curb*3]);
			}
		}
		if((xt-1) >= 0){	
			assign_sub(sum, &target[Wt*3]);
			if(mask[Wt] < 127.0f){
				yb = oy+yt, xb = ox+xt-1;
				yb = (yb < hb) ? yb : hb-1;
				xb = (xb < wb) ? xb : wb-1;
				yb = (yb < 0) ? 0 : yb;
				xb = (xb < 0) ? 0 : xb;
				curb = wb*yb+xb;
				assign_add(sum, &background[curb*3]);
			}
		}
		if((yt+1) < ht){	
			assign_sub(sum, &target[St*3]);
			if(mask[St] < 127.0f){
				yb = oy+yt+1, xb = ox+xt;
				yb = (yb < hb) ? yb : hb-1;
				xb = (xb < wb) ? xb : wb-1;
				yb = (yb < 0) ? 0 : yb;
				xb = (xb < 0) ? 0 : xb;
				curb = wb*yb+xb;
				assign_add(sum, &background[curb*3]);
			}
		}
		if((yt+1) < wt){	
			assign_sub(sum, &target[Et*3]);
			if(mask[Et] < 127.0f){
				yb = oy+yt, xb = ox+xt+1;
				yb = (yb < hb) ? yb : hb-1;
				xb = (xb < wb) ? xb : wb-1;
				yb = (yb < 0) ? 0 : yb;
				xb = (xb < 0) ? 0 : xb;
				curb = wb*yb+xb;
				assign_add(sum, &background[curb*3]);
			}
		}
		fill_boundry(sum, yt, xt, ht, wt, background, wb, hb, oy, ox);
		sum[0] += 4*target[curt*3+0];
		sum[1] += 4*target[curt*3+1];
		sum[2] += 4*target[curt*3+2];
		fixed[curt*3+0] = sum[0];
		fixed[curt*3+1] = sum[1];
		fixed[curt*3+2] = sum[2];
	}	
}
__global__ void	PoissonImageCloningIteration(
	const float *fixed, const float *mask, const float *source, float *target
	,const int wt, const int ht)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	const int Nt = wt*(yt-1)+xt;
	const int Wt = wt*yt+xt-1;
	const int St = wt*(yt+1)+xt;
	const int Et = wt*yt+xt+1;
	float sum[3] = {};
	if(yt < ht and xt < wt){
		assign_add(sum, &fixed[curt*3]);
		if((yt-1) >= 0){ 
			if(mask[Nt] > 127.0f){	
				assign_add(sum, &source[Nt*3]);
			}
		}	
		if((xt-1) >= 0){	
			if(mask[Wt] > 127.0f){	
				assign_add(sum, &source[Wt*3]);
			}
		}	
		if((yt+1) < ht){	
			if(mask[St] > 127.0f){	
				assign_add(sum, &source[St*3]);
			}
		}	
		if((xt+1) < wt){	
			if(mask[Et] > 127.0f){	
				assign_add(sum, &source[Et*3]);
			}
		}	
		target[curt*3+0] = sum[0]/4;
		target[curt*3+1] = sum[1]/4;
		target[curt*3+2] = sum[2]/4;
	}
}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	// set up
	float *fixed, *buf1, *buf2;
	cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf2, 3*wt*ht*sizeof(float));

	// initialize the iteration
	dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);
	CalculateFixed<<<gdim, bdim>>>(
	background, target, mask, fixed,
	wb, hb, wt, ht, oy, ox
	);
	cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);

	 // iterate
	for (int i = 0; i < 10000; ++i) {
		PoissonImageCloningIteration<<<gdim, bdim>>>(
		fixed, mask, buf1, buf2, wt, ht
		);
		PoissonImageCloningIteration<<<gdim, bdim>>>(
		fixed, mask, buf2, buf1, wt, ht
		);
	}
	// copy the image back
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<gdim, bdim>>>(
	background, buf1, mask, output,
	wb, hb, wt, ht, oy, ox
	);

	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);
	
}
