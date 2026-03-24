#ifdef GALAX_MODEL_GPU

#include <cmath>
#include <iostream>

#include "Model_GPU.hpp"
#include "kernel.cuh"


inline bool cuda_malloc(void ** devPtr, size_t size)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc(devPtr, size);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "error: unable to allocate buffer" << std::endl;
		return false;
	}
	return true;
}

inline bool cuda_memcpy(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(dst, src, count, kind);
	if (cudaStatus != cudaSuccess)
	{
		std::cout << "error: unable to copy buffer" << std::endl;
		return false;
	}
	return true;
}

void update_position_gpu(float4* positionsGPU, float4* velocitiesGPU, float* outX,float*  outY,float* outZ,int n_particles,int n_pat_256)
{
    update_position_cu( positionsGPU, velocitiesGPU,
                            outX,  outY, outZ,
                             n_particles,  n_pat_256);
	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		std::cout << "error: unable to synchronize threads" << std::endl;
}


Model_GPU
::Model_GPU(const Initstate& initstate, Particles& particles)
: Model(initstate, particles),
  n_pat_256(((n_particles + 255) / 256) * 256)
{
	// init cuda
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
		std::cout << "error: unable to setup cuda device" << std::endl;

	std::vector<float4> positionsf3(n_pat_256);
	std::vector<float4> velocitiesf3(n_pat_256);

	for (int i = 0; i < n_particles; i++)
	{
		positionsf3[i].x     = initstate.positionsx [i];
		positionsf3[i].y     = initstate.positionsy [i];
		positionsf3[i].z     = initstate.positionsz [i];
		positionsf3[i].w     = initstate.masses[i];
		velocitiesf3[i].x    = initstate.velocitiesx[i];
		velocitiesf3[i].y    = initstate.velocitiesy[i];
		velocitiesf3[i].z    = initstate.velocitiesz[i];
	}
	for (int i = n_particles; i < n_pat_256; i++)
    {
        positionsf3[i] = {0.0f, 0.0f, 0.0f, 0.0f};
        velocitiesf3[i] = {0.0f, 0.0f, 0.0f, 0.0f};
    }

	cuda_malloc((void**)&positionsGPU,     (n_pat_256) * sizeof(float4));

	cuda_malloc((void**)&velocitiesGPU,     (n_pat_256) * sizeof(float4));

	cuda_memcpy(positionsGPU,  positionsf3.data()     , (n_pat_256) * sizeof(float4), cudaMemcpyHostToDevice);

	cuda_memcpy(velocitiesGPU,  velocitiesf3.data()     , (n_pat_256) * sizeof(float4), cudaMemcpyHostToDevice);

	cuda_malloc((void**)&outX, n_particles * sizeof(float));
    cuda_malloc((void**)&outY, n_particles * sizeof(float));
    cuda_malloc((void**)&outZ, n_particles * sizeof(float));
}

Model_GPU
::~Model_GPU()
{
	cudaFree((void**)&positionsGPU);

	cudaFree((void**)&velocitiesGPU);

	cudaFree(outX);
    cudaFree(outY);
    cudaFree(outZ);

}

void Model_GPU
::step()
{
   	update_position_gpu(positionsGPU, velocitiesGPU,outX,  outY, outZ, n_particles,n_pat_256);

    cudaMemcpy(particles.x.data(), outX, n_particles * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.y.data(), outY, n_particles * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(particles.z.data(), outZ, n_particles * sizeof(float), cudaMemcpyDeviceToHost);


}

#endif // GALAX_MODEL_GPU
