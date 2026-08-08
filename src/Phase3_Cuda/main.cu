#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

const int NX = 1000;
const int NY = 1000;
const int NUM_STEPS = 3000;

const double tau = 0.56; 
const double u_inflow = 0.1;

const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
const double w[9] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 
                     1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
const int noslip[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6}; 

__constant__ int d_cx[9];
__constant__ int d_cy[9];
__constant__ double d_w[9];
__constant__ int d_noslip[9];

__global__ void collision_kernel(double* f, const bool* obstacle, double tau_param) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;
    int idx = y * NX + x;

    if (obstacle[idx]) return;

    double rho = 0.0, ux = 0.0, uy = 0.0;
    for (int i = 0; i < 9; ++i) {
        double fi = f[i * NX * NY + idx];
        rho += fi;
        ux  += fi * d_cx[i];
        uy  += fi * d_cy[i];
    }
    ux /= rho;
    uy /= rho;

    double u2 = ux * ux + uy * uy;
    for (int i = 0; i < 9; ++i) {
        double cu = d_cx[i] * ux + d_cy[i] * uy;
        double feq = d_w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        int f_idx = i * NX * NY + idx;
        f[f_idx] = f[f_idx] - (f[f_idx] - feq) / tau_param;
    }
}

__global__ void streaming_kernel(const double* f, double* f_next, const bool* obstacle, double u_inflow_param) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;
    int idx = y * NX + x;

    if (x == 0) {
        double u2 = u_inflow_param * u_inflow_param;
        for (int i = 0; i < 9; ++i) {
            double cu = d_cx[i] * u_inflow_param;
            f_next[i * NX * NY + idx] = d_w[i] * 1.0 * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        }
        return;
    }

    if (x == NX - 1) {
        int virtual_x = NX - 2;
        int v_idx = y * NX + virtual_x;
        for (int i = 0; i < 9; ++i) {
            int prev_x = virtual_x - d_cx[i];
            int prev_y = y - d_cy[i];
            if (prev_y < 0 || prev_y >= NY || obstacle[prev_y * NX + prev_x]) {
                f_next[i * NX * NY + idx] = f[d_noslip[i] * NX * NY + v_idx];
            } else {
                f_next[i * NX * NY + idx] = f[i * NX * NY + (prev_y * NX + prev_x)];
            }
        }
        return;
    }

    if (obstacle[idx]) return;

    for (int i = 0; i < 9; ++i) {
        int prev_x = x - d_cx[i];
        int prev_y = y - d_cy[i];
        int prev_idx = prev_y * NX + prev_x;

        if (prev_y < 0 || prev_y >= NY || obstacle[prev_idx]) {
            f_next[i * NX * NY + idx] = f[d_noslip[i] * NX * NY + idx];
        } else {
            f_next[i * NX * NY + idx] = f[i * NX * NY + prev_idx];
        }
    }
}

int main() {
    std::vector<double> h_f(9 * NX * NY);
    std::vector<char> h_obs(NX * NY, 0);

    int cx_cyl = NX / 4, cy_cyl = NY / 2, r_cyl = NY / 10;
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            if ((x - cx_cyl)*(x - cx_cyl) + (y - cy_cyl)*(y - cy_cyl) < r_cyl * r_cyl) {
                h_obs[y * NX + x] = 1;
            }
        }
    }

    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = y * NX + x;
            double u2_init = u_inflow * u_inflow;
            for (int i = 0; i < 9; ++i) {
                double cu = cx[i] * u_inflow;
                h_f[i * NX * NY + idx] = w[i] * 1.0 * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2_init);
            }
        }
    }

    double *d_f, *d_f_next;
    bool *d_obstacle;
    cudaMalloc(&d_f, 9 * NX * NY * sizeof(double));
    cudaMalloc(&d_f_next, 9 * NX * NY * sizeof(double));
    cudaMalloc(&d_obstacle, NX * NY * sizeof(bool));

    std::vector<char> h_obs_bool(NX * NY);
    for(int i=0; i<NX*NY; ++i) h_obs_bool[i] = (h_obs[i] == 1);

    cudaMemcpy(d_f, h_f.data(), 9 * NX * NY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obstacle, h_obs_bool.data(), NX * NY * sizeof(bool), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_cx, cx, 9 * sizeof(int));
    cudaMemcpyToSymbol(d_cy, cy, 9 * sizeof(int));
    cudaMemcpyToSymbol(d_w, w, 9 * sizeof(double));
    cudaMemcpyToSymbol(d_noslip, noslip, 9 * sizeof(int));

    dim3 blockSize(32, 8);
    dim3 gridSize((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y);

    std::cout << "=========================================================" << std::endl;
    std::cout << ">>> EJECUTANDO FASE 1: SOLO GPU (RTX 3060 Ti)" << std::endl;
    std::cout << "=========================================================" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < NUM_STEPS; ++step) {
        collision_kernel<<<gridSize, blockSize>>>(d_f, d_obstacle, tau);
        streaming_kernel<<<gridSize, blockSize>>>(d_f, d_f_next, d_obstacle, u_inflow);
        std::swap(d_f, d_f_next);
    }
    cudaDeviceSynchronize();

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(end_time - start_time).count();
    double mlups = (double(NX) * NY * NUM_STEPS) / (total_sec * 1e6);

    std::cout << "Tiempo Total Solo GPU: " << total_sec << " segundos." << std::endl;
    std::cout << "Rendimiento Solo GPU : " << mlups << " MLUPS" << std::endl;
    std::cout << "=========================================================" << std::endl;

    cudaFree(d_f); cudaFree(d_f_next); cudaFree(d_obstacle);
    return 0;
}