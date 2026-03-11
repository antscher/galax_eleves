#ifdef GALAX_MODEL_GPU

#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__

#include <stdio.h>

void update_position_cu(float * positionXGPU,float * positionYGPU,float * positionZGPU, float * velocityXGPU,float * velocityYGPU,float * velocityZGPU, float* massesGPU, int n_particles);
#endif

#endif // GALAX_MODEL_GPU
