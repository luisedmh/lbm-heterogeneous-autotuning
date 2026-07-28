#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <cstdlib>
#include <cuda_runtime.h>

// Dimensiones de la malla 3D (160 x 60 x 60 = 576,000 celdas)
const int NX = 160;
const int NY = 60;
const int NZ = 60;
const int NUM_STEPS = 5000;
const int SAVE_EVERY = 100;

const double tau = 0.54;       // Viscosidad
const double u_inflow = 0.05;   // Velocidad de entrada en eje X

// --- VECTORES D3Q19 (19 DIRECCIONES EN 3D) ---
const int cx[19] = { 0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0 };
const int cy[19] = { 0,  0,  0,  1, -1,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1 };
const int cz[19] = { 0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1 };

const double w[19] = {
    12.0/36.0,                                                   // Centro (1)
    2.0/36.0, 2.0/36.0, 2.0/36.0, 2.0/36.0, 2.0/36.0, 2.0/36.0,   // Caras (6)
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,   // Aristas (12)
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};

// Direcciones opuestas para el rebote Bounce-back
const int noslip[19] = { 0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15 };

__constant__ int d_cx[19];
__constant__ int d_cy[19];
__constant__ int d_cz[19];
__constant__ double d_w[19];
__constant__ int d_noslip[19];

// KERNEL GPU 3D: Colisión BGK
__global__ void collision_kernel_3d(double* f, const bool* obstacle, double tau_param) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= NX || y >= NY || z >= NZ) return;
    int idx = z * (NX * NY) + y * NX + x;
    if (obstacle[idx]) return;

    // 1. Densidad y Velocidad 3D
    double rho = 0.0, ux = 0.0, uy = 0.0, uz = 0.0;
    for (int i = 0; i < 19; ++i) {
        double fi = f[i * (NX * NY * NZ) + idx];
        rho += fi;
        ux  += fi * d_cx[i];
        uy  += fi * d_cy[i];
        uz  += fi * d_cz[i];
    }
    ux /= rho; uy /= rho; uz /= rho;

    double u2 = ux * ux + uy * uy + uz * uz;

    // 2. Equilibrio D3Q19 y Colisión
    for (int i = 0; i < 19; ++i) {
        double cu = d_cx[i] * ux + d_cy[i] * uy + d_cz[i] * uz;
        double feq = d_w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        int f_idx = i * (NX * NY * NZ) + idx;
        f[f_idx] = f[f_idx] - (f[f_idx] - feq) / tau_param;
    }
}

// KERNEL GPU 3D: Streaming PULL con Condiciones de Contorno
__global__ void streaming_kernel_3d(const double* f, double* f_next, const bool* obstacle, double u_in) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= NX || y >= NY || z >= NZ) return;
    int idx = z * (NX * NY) + y * NX + x;

    // Entrada (x = 0): Flujo constante en X
    if (x == 0) {
        double u2 = u_in * u_in;
        for (int i = 0; i < 19; ++i) {
            double cu = d_cx[i] * u_in;
            f_next[i * (NX * NY * NZ) + idx] = d_w[i] * 1.0 * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        }
        return;
    }

    // Salida (x = NX - 1): Copia del plano anterior
    if (x == NX - 1) {
        int prev_idx = z * (NX * NY) + y * NX + (NX - 2);
        for (int i = 0; i < 19; ++i) {
            f_next[i * (NX * NY * NZ) + idx] = f[i * (NX * NY * NZ) + prev_idx];
        }
        return;
    }

    if (obstacle[idx]) return;

    // Streaming normal interior
    for (int i = 0; i < 19; ++i) {
        int prev_x = x - d_cx[i];
        int prev_y = y - d_cy[i];
        int prev_z = z - d_cz[i];

        // Rebote en las paredes externas del túnel (Y y Z)
        if (prev_y < 0 || prev_y >= NY || prev_z < 0 || prev_z >= NZ) {
            f_next[i * (NX * NY * NZ) + idx] = f[d_noslip[i] * (NX * NY * NZ) + idx];
        } 
        // Rebote en la esfera sólida
        else {
            int prev_idx = prev_z * (NX * NY) + prev_y * NX + prev_x;
            if (obstacle[prev_idx]) {
                f_next[i * (NX * NY * NZ) + idx] = f[d_noslip[i] * (NX * NY * NZ) + idx];
            } else {
                f_next[i * (NX * NY * NZ) + idx] = f[i * (NX * NY * NZ) + prev_idx];
            }
        }
    }
}

void save_vtk_3d(int step, const std::vector<double>& h_f, const std::vector<bool>& h_obstacle);

int main() {
    int total_cells = NX * NY * NZ;
    std::vector<double> h_f(19 * total_cells);
    std::vector<bool> h_obstacle(total_cells, false);

    // Obstáculo: Esfera en 3D situada en (NX/4, NY/2, NZ/2)
    int cx_sph = NX / 4, cy_sph = NY / 2, cz_sph = NZ / 2;
    int r_sph = NY / 6;

    for (int z = 0; z < NZ; ++z) {
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                int idx = z * (NX * NY) + y * NX + x;
                if ((x - cx_sph)*(x - cx_sph) + (y - cy_sph)*(y - cy_sph) + (z - cz_sph)*(z - cz_sph) < r_sph * r_sph) {
                    h_obstacle[idx] = true;
                }
            }
        }
    }

    // Inicializar fluido en reposo/equilibrio
    for (int idx = 0; idx < total_cells; ++idx) {
        double u2 = u_inflow * u_inflow;
        for (int i = 0; i < 19; ++i) {
            double cu = cx[i] * u_inflow;
            h_f[i * total_cells + idx] = w[i] * 1.0 * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        }
    }

    std::vector<char> h_obs_char(total_cells);
    for (int i = 0; i < total_cells; ++i) h_obs_char[i] = h_obstacle[i] ? 1 : 0;

    // Memoria GPU
    double *d_f, *d_f_next;
    bool *d_obstacle;
    cudaMalloc(&d_f, 19 * total_cells * sizeof(double));
    cudaMalloc(&d_f_next, 19 * total_cells * sizeof(double));
    cudaMalloc(&d_obstacle, total_cells * sizeof(bool));

    cudaMemcpy(d_f, h_f.data(), 19 * total_cells * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obstacle, h_obs_char.data(), total_cells * sizeof(bool), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_cx, cx, 19 * sizeof(int));
    cudaMemcpyToSymbol(d_cy, cy, 19 * sizeof(int));
    cudaMemcpyToSymbol(d_cz, cz, 19 * sizeof(int));
    cudaMemcpyToSymbol(d_w, w, 19 * sizeof(double));
    cudaMemcpyToSymbol(d_noslip, noslip, 19 * sizeof(int));

    // Mapeo de hilos 3D (Bloques de 8 x 8 x 4 = 256 hilos por bloque)
    dim3 blockSize(8, 8, 4);
    dim3 gridSize((NX + blockSize.x - 1) / blockSize.x,
                 (NY + blockSize.y - 1) / blockSize.y,
                 (NZ + blockSize.z - 1) / blockSize.z);

    std::cout << "Iniciando Simulación 3D LBM (D3Q19) en CUDA..." << std::endl;
    std::cout << "Tamaño de la malla 3D: " << NX << " x " << NY << " x " << NZ << " (" << total_cells << " celdas)" << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int step = 0; step <= NUM_STEPS; ++step) {
        collision_kernel_3d<<<gridSize, blockSize>>>(d_f, d_obstacle, tau);
        streaming_kernel_3d<<<gridSize, blockSize>>>(d_f, d_f_next, d_obstacle, u_inflow);
        std::swap(d_f, d_f_next);

        if (step % SAVE_EVERY == 0) {
            std::cout << "Paso GPU 3D: " << step << " / " << NUM_STEPS << std::endl;
            cudaMemcpy(h_f.data(), d_f, 19 * total_cells * sizeof(double), cudaMemcpyDeviceToHost);
            save_vtk_3d(step, h_f, h_obstacle);
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "=========================================================" << std::endl;
    std::cout << "¡Simulación 3D Completada!" << std::endl;
    std::cout << "Tiempo de ejecución en GPU: " << (ms / 1000.0) << " segundos." << std::endl;
    std::cout << "=========================================================" << std::endl;

    cudaFree(d_f); cudaFree(d_f_next); cudaFree(d_obstacle);
    return 0;
}

// Generador de Archivos VTK 3D para ParaView
void save_vtk_3d(int step, const std::vector<double>& h_f, const std::vector<bool>& h_obstacle) {
    namespace fs = std::filesystem;
    fs::path out_dir = fs::path(std::getenv("HOME")) / "Documents/Proyectos/Informática/HPC/LMB_Autotuning/results/phase5_3d";
    fs::create_directories(out_dir);

    std::stringstream ss;
    ss << "fluid3d_" << std::setw(4) << std::setfill('0') << step << ".vtk";
    std::ofstream out((out_dir / ss.str()).string());

    int total_cells = NX * NY * NZ;

    out << "# vtk DataFile Version 3.0\nLBM 3D D3Q19 GPU\nASCII\nDATASET STRUCTURED_POINTS\n";
    out << "DIMENSIONS " << NX << " " << NY << " " << NZ << "\nORIGIN 0 0 0\nSPACING 1 1 1\n";
    out << "POINT_DATA " << total_cells << "\nSCALARS velocity_magnitude double 1\nLOOKUP_TABLE default\n";

    for (int z = 0; z < NZ; ++z) {
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                int idx = z * (NX * NY) + y * NX + x;
                if (h_obstacle[idx]) {
                    out << 0.0 << "\n";
                } else {
                    double rho = 0, ux = 0, uy = 0, uz = 0;
                    for (int i = 0; i < 19; ++i) {
                        double fi = h_f[i * total_cells + idx];
                        rho += fi;
                        ux  += fi * cx[i];
                        uy  += fi * cy[i];
                        uz  += fi * cz[i];
                    }
                    out << std::sqrt((ux/rho)*(ux/rho) + (uy/rho)*(uy/rho) + (uz/rho)*(uz/rho)) << "\n";
                }
            }
        }
    }
}