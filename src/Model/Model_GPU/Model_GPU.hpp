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
	std::vector<float> velocityXf   ;
	std::vector<float> velocityYf   ;
	std::vector<float> velocityZf   ;

	float*  positionXGPU;
	float*  positionYGPU;
	float*  positionZGPU;
	float*  velocityXGPU;
	float*  velocityYGPU;
	float*  velocityZGPU;
	float*  massesGPU;

public:
	Model_GPU(const Initstate& initstate, Particles& particles);

	virtual ~Model_GPU();

	virtual void step();
};
#endif // MODEL_GPU_HPP_

#endif // GALAX_MODEL_GPU
