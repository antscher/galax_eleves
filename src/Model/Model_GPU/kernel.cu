#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)
#define BLOCK_SIZE 256

__global__ void compute_acc(float4 * positionsGPU, float4 * velocitiesGPU, int diff, int n_pat_256)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pat_256) return;

    // (Double Buffering setup)
    __shared__ float4 tile[2][BLOCK_SIZE];

    float4 p_i = positionsGPU[i];
    float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};

    int num_tiles = n_pat_256 / BLOCK_SIZE;

    // 1. Pre-load the first tile
    tile[0][threadIdx.x] = positionsGPU[threadIdx.x];
    __syncthreads();

    int current = 0; // Index of the tile we are computing
    int next = 1;    // Index of the tile we are loading

    for (int t = 0; t < num_tiles -1; t++) {

        // 2. Start loading the next tile (if it exists)
        int j_global = (t + 1) * BLOCK_SIZE + threadIdx.x;
        tile[next][threadIdx.x] = positionsGPU[j_global];

        // 3. Compute interactions using the current tile
        //#pragma unroll 4
        for (int j = 0; j < BLOCK_SIZE; j++) {
            float4 p_j = tile[current][j];

            const float diffx = p_j.x - p_i.x;
            const float diffy = p_j.y - p_i.y;
            const float diffz = p_j.z - p_i.z;

            float dij = diffx * diffx + diffy * diffy + diffz * diffz;

            //Removing the condition if
            float max_dij = fmaxf(dij, 1.0f);
            float inv_dist = rsqrtf(max_dij);
            float factor = 10.0f * (inv_dist * inv_dist * inv_dist);

            acc.x = fmaf(diffx, factor * p_j.w, acc.x);
            acc.y = fmaf(diffy, factor * p_j.w, acc.y);
            acc.z = fmaf(diffz, factor * p_j.w, acc.z);
        }


        // 4. Wait for both the computation and the loading of the next tile to finish
        __syncthreads();

        // Swap the roles for the next iteration
        current = 1 - current;
        next = 1 - next;
    }
    for (int j = 0; j < diff; j++) {
        float4 p_j = tile[current][j];

        const float diffx = p_j.x - p_i.x;
        const float diffy = p_j.y - p_i.y;
        const float diffz = p_j.z - p_i.z;

        float dij = diffx * diffx + diffy * diffy + diffz * diffz;

        //Removing the condition if
        float max_dij = fmaxf(dij, 1.0f);
        float inv_dist = rsqrtf(max_dij);
        float factor = 10.0f * (inv_dist * inv_dist * inv_dist);

        acc.x = fmaf(diffx, factor * p_j.w, acc.x);
        acc.y = fmaf(diffy, factor * p_j.w, acc.y);
        acc.z = fmaf(diffz, factor * p_j.w, acc.z);
    }
    __syncthreads();


    velocitiesGPU[i].x = fmaf(acc.x, 2.0f, velocitiesGPU[i].x);
    velocitiesGPU[i].y = fmaf(acc.y, 2.0f, velocitiesGPU[i].y);
    velocitiesGPU[i].z = fmaf(acc.z, 2.0f, velocitiesGPU[i].z);

}

__global__ void maj_pos(float4 * positionsGPU, float4 * velocitiesGPU,
                        float * outX, float * outY, float * outZ,
                        int n_particles)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned int)n_particles) return;

    float4 vel = velocitiesGPU[i];

    positionsGPU[i].x = fmaf(vel.x, 0.1f, positionsGPU[i].x);
    positionsGPU[i].y = fmaf(vel.y, 0.1f, positionsGPU[i].y);
    positionsGPU[i].z = fmaf(vel.z, 0.1f, positionsGPU[i].z);

    outX[i] = positionsGPU[i].x;
    outY[i] = positionsGPU[i].y;
    outZ[i] = positionsGPU[i].z;
}

void update_position_cu(float4* positionsGPU, float4* velocitiesGPU,
                        float* outX, float* outY, float* outZ,
                        int n_particles, int n_pat_256)
{
    int nthreads = 256;
    int nblocks =  (n_particles + (nthreads -1)) / nthreads;

    compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU,256 - n_pat_256 + n_particles, n_pat_256);

    maj_pos<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, outX, outY, outZ, n_particles);
}

#endif // GALAX_MODEL_GPU
