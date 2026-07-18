#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <cstdlib>
#include <omp.h>
#include <cuda_runtime.h>

// Dimensiones globales
const int NX = 400;
const int NY = 100;
const int NUM_STEPS = 8000;
const int SAVE_EVERY = 100;

// --- CONFIGURACIÓN ASIMÉTRICA OPTIMIZADA ---
const int SPLIT_X = 50; // CPU controla de 0 a 49. GPU controla de 50 a 399.

// Parámetros físicos
const double tau = 0.56; 
const double u_inflow = 0.1;

// Constantes en Host
const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
const double w[9] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 
                     1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
const int noslip[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6}; 

// Constantes en Device (GPU)
__constant__ int d_cx[9];
__constant__ int d_cy[9];
__constant__ double d_w[9];
__constant__ int d_noslip[9];

// --- KERNELS GPU (CUDA) ---

__global__ void collision_kernel_gpu(double* f, const bool* obstacle, double tau_param, int split_x) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;
    if (x < split_x) return; // La GPU ignora el territorio de la CPU

    int idx = y * NX + x;
    if (obstacle[idx]) return;

    double rho = 0.0, ux = 0.0, uy = 0.0;
    for (int i = 0; i < 9; ++i) {
        double fi = f[i * NX * NY + idx];
        rho += fi; ux += fi * d_cx[i]; uy += fi * d_cy[i];
    }
    ux /= rho; uy /= rho;

    double u2 = ux * ux + uy * uy;
    for (int i = 0; i < 9; ++i) {
        double cu = d_cx[i] * ux + d_cy[i] * uy;
        double feq = d_w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        int f_idx = i * NX * NY + idx;
        f[f_idx] = f[f_idx] - (f[f_idx] - feq) / tau_param;
    }
}

__global__ void streaming_kernel_gpu(const double* f, double* f_next, const bool* obstacle, int split_x) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;
    if (x < split_x) return; // La GPU ignora el territorio de la CPU

    int idx = y * NX + x;

    // Condición de contorno de Salida (Extremo Derecho) integrada en GPU
    if (x == NX - 1) {
        int v_idx = y * NX + (NX - 2);
        for (int i = 0; i < 9; ++i) {
            int prev_x = (NX - 2) - d_cx[i];
            int prev_y = y - d_cy[i];
            if (prev_y < 0 || prev_y >= NY) f_next[i * NX * NY + idx] = f[d_noslip[i] * NX * NY + v_idx];
            else if (obstacle[prev_y * NX + prev_x]) f_next[i * NX * NY + idx] = f[d_noslip[i] * NX * NY + v_idx];
            else f_next[i * NX * NY + idx] = f[i * NX * NY + (prev_y * NX + prev_x)];
        }
        return;
    }

    if (obstacle[idx]) return;

    // Streaming normal GPU
    for (int i = 0; i < 9; ++i) {
        int prev_x = x - d_cx[i];
        int prev_y = y - d_cy[i];
        int prev_idx = prev_y * NX + prev_x;

        if (prev_y < 0 || prev_y >= NY) f_next[i * NX * NY + idx] = f[d_noslip[i] * NX * NY + idx];
        else if (obstacle[prev_idx])     f_next[i * NX * NY + idx] = f[d_noslip[i] * NX * NY + idx];
        else                            f_next[i * NX * NY + idx] = f[i * NX * NY + prev_idx];
    }
}

inline double equilibrium_cpu(int i, double rho, double ux, double uy) {
    double cu = cx[i] * ux + cy[i] * uy;
    double u2 = ux * ux + uy * uy;
    return w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
}

void save_vtk(int step, const std::vector<double>& h_f, const std::vector<bool>& h_obstacle);

int main() {
    std::vector<double> h_f(9 * NX * NY);
    std::vector<double> h_f_next(9 * NX * NY);
    std::vector<bool> h_obstacle(NX * NY, false);

    // Definición del cilindro (Cae en x=100, territorio de la GPU)
    int cx_cyl = NX / 4, cy_cyl = NY / 2, r_cyl = NY / 10;
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            if ((x - cx_cyl)*(x - cx_cyl) + (y - cy_cyl)*(y - cy_cyl) < r_cyl * r_cyl) {
                h_obstacle[y * NX + x] = true;
            }
        }
    }

    // Inicializar mapa
    for (int idx = 0; idx < NX * NY; ++idx) {
        for (int i = 0; i < 9; ++i) {
            h_f[i * NX * NY + idx] = equilibrium_cpu(i, 1.0, u_inflow, 0.0);
        }
    }

    std::vector<char> h_obs_char(NX * NY);
    for(int i=0; i<NX*NY; ++i) h_obs_char[i] = h_obstacle[i] ? 1 : 0;

    // Memoria GPU
    double *d_f, *d_f_next;
    bool *d_obstacle;
    cudaMalloc(&d_f, 9 * NX * NY * sizeof(double));
    cudaMalloc(&d_f_next, 9 * NX * NY * sizeof(double));
    cudaMalloc(&d_obstacle, NX * NY * sizeof(bool));

    cudaMemcpy(d_f, h_f.data(), 9 * NX * NY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obstacle, h_obs_char.data(), NX * NY * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_cx, cx, 9 * sizeof(int));
    cudaMemcpyToSymbol(d_cy, cy, 9 * sizeof(int));
    cudaMemcpyToSymbol(d_w, w, 9 * sizeof(double));
    cudaMemcpyToSymbol(d_noslip, noslip, 9 * sizeof(int));

    dim3 blockSize(32, 8);
    dim3 gridSize((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y);

    std::cout << "Corriendo Distribución Híbrida Asimétrica (50 CPU / 350 GPU)..." << std::endl;
    double start_time = omp_get_wtime();

    for (int step = 0; step <= NUM_STEPS; ++step) {
        
        // =====================================================================
        // PASO 1: COLISIÓN SIMULTÁNEA
        // =====================================================================
        collision_kernel_gpu<<<gridSize, blockSize>>>(d_f, d_obstacle, tau, SPLIT_X);

        // CPU procesa su franja (0 a 49)
        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < SPLIT_X; ++x) {
                int idx = y * NX + x;
                double rho = 0, ux = 0, uy = 0;
                for (int i = 0; i < 9; ++i) {
                    double fi = h_f[i * NX * NY + idx];
                    rho += fi; ux += fi * cx[i]; uy += fi * cy[i];
                }
                ux /= rho; uy /= rho;
                for (int i = 0; i < 9; ++i) {
                    int f_idx = i * NX * NY + idx;
                    h_f[f_idx] = h_f[f_idx] - (h_f[f_idx] - equilibrium_cpu(i, rho, ux, uy)) / tau;
                }
            }
        }

        cudaDeviceSynchronize();

        // =====================================================================
        // PASO 2: UNICO INTERCAMBIO DE HALO DE FRONTERA (Frontera x = 50)
        // =====================================================================
        for (int i = 0; i < 9; ++i) {
            int offset = i * NX * NY;
            // CPU col 49 -> GPU col 49 (Para que el streaming de la GPU lea de la izquierda)
            cudaMemcpy2D(&d_f[offset + (SPLIT_X - 1)], NX * sizeof(double), &h_f[offset + (SPLIT_X - 1)], NX * sizeof(double), sizeof(double), NY, cudaMemcpyHostToDevice);
            // GPU col 50 -> CPU col 50 (Para que el streaming de la CPU lea de la derecha)
            cudaMemcpy2D(&h_f[offset + SPLIT_X], NX * sizeof(double), &d_f[offset + SPLIT_X], NX * sizeof(double), sizeof(double), NY, cudaMemcpyDeviceToHost);
        }

        // =====================================================================
        // PASO 3: STREAMING SIMULTÁNEO
        // =====================================================================
        streaming_kernel_gpu<<<gridSize, blockSize>>>(d_f, d_f_next, d_obstacle, SPLIT_X);

        // CPU Streaming (0 a 49)
        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < SPLIT_X; ++x) {
                int idx = y * NX + x;
                if (x == 0) { // Entrada Inflow fija en el extremo CPU
                    for (int i = 0; i < 9; ++i) h_f_next[i * NX * NY + idx] = equilibrium_cpu(i, 1.0, u_inflow, 0.0);
                    continue;
                }
                for (int i = 0; i < 9; ++i) {
                    int prev_x = x - cx[i], prev_y = y - cy[i];
                    if (prev_y < 0 || prev_y >= NY) h_f_next[i * NX * NY + idx] = h_f[noslip[i] * NX * NY + idx];
                    else                           h_f_next[i * NX * NY + idx] = h_f[i * NX * NY + (prev_y * NX + prev_x)];
                }
            }
        }

        cudaDeviceSynchronize();

        std::swap(h_f, h_f_next);
        std::swap(d_f, d_f_next);

        // =====================================================================
        // PASO 4: RECONSTRUCCIÓN VTK
        // =====================================================================
        if (step % SAVE_EVERY == 0) {
            std::cout << "Paso Híbrido 50/350: " << step << " / " << NUM_STEPS << std::endl;
            // Descargar la ventana completa de la GPU (de la col 50 a la 399)
            for (int i = 0; i < 9; ++i) {
                cudaMemcpy2D(
                    &h_f[i * NX * NY + SPLIT_X], NX * sizeof(double),
                    &d_f[i * NX * NY + SPLIT_X], NX * sizeof(double),
                    (NX - SPLIT_X) * sizeof(double), NY, cudaMemcpyDeviceToHost
                );
            }
            save_vtk(step, h_f, h_obstacle);
        }
    }

    double end_time = omp_get_wtime();
    std::cout << "=========================================================" << std::endl;
    std::cout << "¡Simulación Híbrida 50/350 Completada!" << std::endl;
    std::cout << "Tiempo de ejecución acoplado: " << (end_time - start_time) << " segundos." << std::endl;
    std::cout << "=========================================================" << std::endl;

    cudaFree(d_f); cudaFree(d_f_next); cudaFree(d_obstacle);
    return 0;
}

void save_vtk(int step, const std::vector<double>& h_f, const std::vector<bool>& h_obstacle) {
    namespace fs = std::filesystem;
    fs::path out_dir = fs::path(std::getenv("HOME")) / "Documents/Proyectos/Informática/HPC/LMB_Autotuning/results/phase4";
    fs::create_directories(out_dir);
    std::stringstream ss;
    ss << "fluid_" << std::setw(4) << std::setfill('0') << step << ".vtk";
    std::ofstream out((out_dir / ss.str()).string());

    out << "# vtk DataFile Version 3.0\nLBM D2Q9 Hibrido 50-350\nASCII\nDATASET STRUCTURED_POINTS\n";
    out << "DIMENSIONS " << NX << " " << NY << " 1\nORIGIN 0 0 0\nSPACING 1 1 1\n";
    out << "POINT_DATA " << NX * NY << "\nSCALARS velocity_magnitude double 1\nLOOKUP_TABLE default\n";

    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = y * NX + x;
            if (h_obstacle[idx]) out << 0.0 << "\n";
            else {
                double rho = 0, ux = 0, uy = 0;
                for (int i = 0; i < 9; ++i) {
                    double fi = h_f[i * NX * NY + idx];
                    rho += fi; ux += fi * cx[i]; uy += fi * cy[i];
                }
                out << std::sqrt((ux/rho)*(ux/rho) + (uy/rho)*(uy/rho)) << "\n";
            }
        }
    }
    out << "VECTORS velocity double\n";
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = y * NX + x;
            if (h_obstacle[idx]) out << "0.0 0.0 0.0\n";
            else {
                double rho = 0, ux = 0, uy = 0;
                for (int i = 0; i < 9; ++i) {
                    double fi = h_f[i * NX * NY + idx];
                    rho += fi; ux += fi * cx[i]; uy += fi * cy[i];
                }
                out << ux/rho << " " << uy/rho << " 0.0\n";
            }
        }
    }
}