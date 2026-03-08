#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)

__global__ void compute_acc(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, float* massesGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= (unsigned int)n_particles) return;

	accelerationsGPU[i].x = 0.0f;
	accelerationsGPU[i].y = 0.0f;
	accelerationsGPU[i].z = 0.0f;

	for (int j = 0; j < n_particles; j++)
	{
		if(i == j)continue;

		const float diffx = positionsGPU[j].x - positionsGPU[i].x;
		const float diffy = positionsGPU[j].y - positionsGPU[i].y;
		const float diffz = positionsGPU[j].z - positionsGPU[i].z;

		float dij = diffx * diffx + diffy * diffy + diffz * diffz;

		if (dij < 1.0)  // ← DIVERGENCE : split dans les threads
        {
     			dij = 10.0;
        }
        else
        {
     			dij = std::sqrt(dij);  // Opération coûteuse !
     			dij = 10.0 / (dij * dij * dij);
        }

		accelerationsGPU[i].x += diffx * dij * massesGPU[j];
		accelerationsGPU[i].y += diffy * dij * massesGPU[j];
		accelerationsGPU[i].z += diffz * dij * massesGPU[j];
	}
}

__global__ void maj_pos(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= (unsigned int)n_particles) return;

	velocitiesGPU[i].x += accelerationsGPU[i].x * 2.0f;
	velocitiesGPU[i].y += accelerationsGPU[i].y * 2.0f;
	velocitiesGPU[i].z += accelerationsGPU[i].z * 2.0f;
	positionsGPU[i].x += velocitiesGPU[i].x * 0.1f;
	positionsGPU[i].y += velocitiesGPU[i].y * 0.1f;
	positionsGPU[i].z += velocitiesGPU[i].z * 0.1f;


}

void update_position_cu(float3* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, float* massesGPU, int n_particles)
{
	int nthreads = 256;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, massesGPU, n_particles);

	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}


#endif // GALAX_MODEL_GPU
