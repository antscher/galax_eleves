#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)
#define BLOCK_SIZE 256

__global__ void compute_acc(float4 * positionsGPU, float4 * velocitiesGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= (unsigned int)n_particles) return;

	__shared__ float4 tile[BLOCK_SIZE];

	float4 p_i = positionsGPU[i];
	float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

	int num_tiles = n_particles / BLOCK_SIZE;

	for (int t = 0; t < num_tiles; t++) {
		//if(i == j)continue;
		// Chargement
		int j_global = t * BLOCK_SIZE + threadIdx.x;
        tile[threadIdx.x] = positionsGPU[j_global];

        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; j++) {
            float4 p_j = tile[j];
    		const float diffx = p_j.x - p_i.x;
    		const float diffy = p_j.y - p_i.y;
    		const float diffz = p_j.z - p_i.z;

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

    		acc.x += diffx * dij * p_j.w;
    		acc.y += diffy * dij * p_j.w;
    		acc.z += diffz * dij * p_j.w;
        }
        //Wait for end of calcul block
        __syncthreads();
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

void update_position_cu(float4* positionsGPU, float4* velocitiesGPU, int n_particles, int n_pat_256)
{
	int nthreads = 256;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, n_pat_256);

	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, n_particles);
}


#endif // GALAX_MODEL_GPU
