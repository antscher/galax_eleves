#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)

__global__ void compute_acc(float4 * positionsGPU, float4 * velocitiesGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= (unsigned int)n_particles) return;

	float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

	for (int j = 0; j < n_particles; j++)
	{
		//if(i == j)continue;

		const float diffx = positionsGPU[j].x - positionsGPU[i].x;
		const float diffy = positionsGPU[j].y - positionsGPU[i].y;
		const float diffz = positionsGPU[j].z - positionsGPU[i].z;

		float dij = diffx * diffx + diffy * diffy + diffz * diffz;

		if (dij < 1.0f)
        {
     			dij = 10.0f;
        }
        else
        {
     			dij = rsqrtf(dij);  // Opération coûteuse !
     			dij = 10.0f * (dij * dij * dij);
        }

		acc.x += diffx * dij * positionsGPU[j].w;
		acc.y += diffy * dij * positionsGPU[j].w;
		acc.z += diffz * dij * positionsGPU[j].w;


	}
	velocitiesGPU[i].x = fmaf(acc.x , 2.0f, velocitiesGPU[i].x);
	velocitiesGPU[i].y = fmaf(acc.y , 2.0f, velocitiesGPU[i].y);
	velocitiesGPU[i].z = fmaf(acc.z , 2.0f, velocitiesGPU[i].z);

}

__global__ void maj_pos(float4 * positionsGPU, float4 * velocitiesGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= (unsigned int)n_particles) return;

	positionsGPU[i].x = fmaf(velocitiesGPU[i].x, 0.1f, positionsGPU[i].x);
	positionsGPU[i].y = fmaf(velocitiesGPU[i].y, 0.1f, positionsGPU[i].y);
	positionsGPU[i].z = fmaf(velocitiesGPU[i].z, 0.1f, positionsGPU[i].z);

}

void update_position_cu(float4* positionsGPU, float4* velocitiesGPU, int n_particles)
{
	int nthreads = 256;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, n_particles);

	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, n_particles);
}


#endif // GALAX_MODEL_GPU
