#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)

__global__ void compute_acc(float * positionXGPU,float * positionYGPU,float * positionZGPU, float * velocityXGPU,float * velocityYGPU,float * velocityZGPU, float* massesGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= (unsigned int)n_particles) return;

	float3 acc = {0.0f, 0.0f, 0.0f};

	for (int j = 0; j < n_particles; j++)
	{
		if(i == j)continue;

		const float diffx = positionXGPU[j] - positionXGPU[i];
		const float diffy = positionYGPU[j] - positionYGPU[i];
		const float diffz = positionZGPU[j] - positionZGPU[i];

		float dij = diffx * diffx + diffy * diffy + diffz * diffz;

		if (dij < 1.0)
        {
     			dij = 10.0;
        }
        else
        {
     			dij = std::sqrt(dij);
     			dij = 10.0 / (dij * dij * dij);
        }

		acc.x += diffx * dij * massesGPU[j];
		acc.y += diffy * dij * massesGPU[j];
		acc.z += diffz * dij * massesGPU[j];


	}
	velocityXGPU[i] += acc.x * 2.0f;
	velocityYGPU[i] += acc.y * 2.0f;
	velocityZGPU[i] += acc.z * 2.0f;

}

__global__ void maj_pos(float * positionXGPU,float * positionYGPU,float * positionZGPU,  float * velocityXGPU,float * velocityYGPU,float * velocityZGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= (unsigned int)n_particles) return;

	positionXGPU[i] += velocityXGPU[i] * 0.1f;
	positionYGPU[i] += velocityYGPU[i] * 0.1f;
	positionZGPU[i] += velocityZGPU[i] * 0.1f;


}

void update_position_cu(float * positionXGPU,float * positionYGPU,float * positionZGPU,  float * velocityXGPU,float * velocityYGPU,float * velocityZGPU, float* massesGPU, int n_particles)
{
	int nthreads = 256;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc<<<nblocks, nthreads>>>(positionXGPU, positionYGPU, positionZGPU,velocityXGPU, velocityYGPU, velocityZGPU, massesGPU, n_particles);

	maj_pos    <<<nblocks, nthreads>>>(positionXGPU, positionYGPU, positionZGPU, velocityXGPU, velocityYGPU, velocityZGPU, n_particles);
}


#endif // GALAX_MODEL_GPU
