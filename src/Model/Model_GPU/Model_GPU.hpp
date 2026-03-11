#ifdef GALAX_MODEL_GPU

#ifndef MODEL_GPU_HPP_
#define MODEL_GPU_HPP_

#include "../Model.hpp"

#include <cuda_runtime.h>
#include "kernel.cuh"

class Model_GPU : public Model
{
private:

	std::vector<float> positionXf    ;
	std::vector<float> positionYf    ;
	std::vector<float> positionZf    ;
	std::vector<float3> velocitiesf3   ;
	std::vector<float3> accelerationsf3;

	float*  positionXGPU;
	float*  positionYGPU;
	float*  positionZGPU;
	float3* velocitiesGPU;
	float*  massesGPU;

public:
	Model_GPU(const Initstate& initstate, Particles& particles);

	virtual ~Model_GPU();

	virtual void step();
};
#endif // MODEL_GPU_HPP_

#endif // GALAX_MODEL_GPU
