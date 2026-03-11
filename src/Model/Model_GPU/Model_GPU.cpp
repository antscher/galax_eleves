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

void update_position_gpu(float * positionXGPU,float * positionYGPU,float * positionZGPU, float3* velocitiesGPU, float* massesGPU, int n_particles)
{
	update_position_cu(positionXGPU, positionYGPU, positionZGPU, velocitiesGPU, massesGPU, n_particles);
	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		std::cout << "error: unable to synchronize threads" << std::endl;
}


Model_GPU
::Model_GPU(const Initstate& initstate, Particles& particles)
: Model(initstate, particles),
  positionXf    (n_particles),
  positionYf    (n_particles),
  positionZf    (n_particles),
  velocitiesf3   (n_particles),
  accelerationsf3(n_particles)
{
	// init cuda
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
		std::cout << "error: unable to setup cuda device" << std::endl;

	for (int i = 0; i < n_particles; i++)
	{
		positionXf[i]    = initstate.positionsx [i];
		positionYf[i]     = initstate.positionsy [i];
		positionZf[i]     = initstate.positionsz [i];
		velocitiesf3[i].x    = initstate.velocitiesx[i];
		velocitiesf3[i].y    = initstate.velocitiesy[i];
		velocitiesf3[i].z    = initstate.velocitiesz[i];
	}

	cuda_malloc((void**)&positionXGPU,     n_particles * sizeof(float));
	cuda_malloc((void**)&positionYGPU,     n_particles * sizeof(float));
	cuda_malloc((void**)&positionZGPU,     n_particles * sizeof(float));

	cuda_malloc((void**)&velocitiesGPU,     n_particles * sizeof(float3));


	cuda_malloc((void**)&massesGPU,     n_particles * sizeof(float3));


	cuda_memcpy(positionXGPU,  positionXf.data()     , n_particles * sizeof(float), cudaMemcpyHostToDevice);
	cuda_memcpy(positionYGPU,  positionYf.data()     , n_particles * sizeof(float), cudaMemcpyHostToDevice);
	cuda_memcpy(positionZGPU,  positionZf.data()     , n_particles * sizeof(float), cudaMemcpyHostToDevice);

	cuda_memcpy(velocitiesGPU,  velocitiesf3.data()     , n_particles * sizeof(float3), cudaMemcpyHostToDevice);

	cuda_memcpy(massesGPU,  initstate.masses.data()     , n_particles * sizeof(float), cudaMemcpyHostToDevice);

}

Model_GPU
::~Model_GPU()
{
	cudaFree((void**)&positionXGPU);
	cudaFree((void**)&positionYGPU);
	cudaFree((void**)&positionZGPU);

	cudaFree((void**)&velocitiesGPU);


	cudaFree((void**)&massesGPU);
}

void Model_GPU
::step()
{
   	update_position_gpu(positionXGPU, positionYGPU, positionZGPU, velocitiesGPU, massesGPU, n_particles);

	cuda_memcpy(positionXf.data(), positionXGPU, n_particles * sizeof(float), cudaMemcpyDeviceToHost);
	cuda_memcpy(positionYf.data(), positionYGPU, n_particles * sizeof(float), cudaMemcpyDeviceToHost);
	cuda_memcpy(positionZf.data(), positionZGPU, n_particles * sizeof(float), cudaMemcpyDeviceToHost);


	for (int i = 0; i < n_particles; i++)
	{
		particles.x[i] = positionXf[i];
		particles.y[i] = positionYf[i];
		particles.z[i] = positionZf[i];
	}


}

#endif // GALAX_MODEL_GPU
