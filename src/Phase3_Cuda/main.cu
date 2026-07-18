#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <cstdlib>
#include <cuda.h> // <--- Librería nativa de CUDA Runtime

// Dimensiones del dominio
const int NX = 400;
const int NY = 100;
const int NUM_STEPS = 8000;
const int SAVE_EVERY = 100;

// Parámetros físicos
const double tau = 0.56; 
const double u_inflow = 0.1;

// Vectores de dirección en el Host (CPU)
const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
const double w[9] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 
                     1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
const int noslip[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6}; 

// --- MEMORIA CONSTANTE DE LA GPU ---
// Colocar estas variables aquí hace que la GPU las lea a velocidad ultra-rápida (L1 Cache)
__constant__ int d_cx[9];
__constant__ int d_cy[9];
__constant__ double d_w[9];
__constant__ int d_noslip[9];

// KERNEL 1: Colisión Local en GPU (Estructura SoA)
__global__ void collision_kernel(double* f, const bool* obstacle, double tau_param) {
    // Calcular las coordenadas (x, y) únicas de este hilo en la rejilla GPU
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Controlar que el hilo no se salga del mapa de simulación
    if (x >= NX || y >= NY) return;
    int idx = y * NX + x;

    if (obstacle[idx]) return;

    // Calcular macroscópicos
    double rho = 0.0;
    double ux = 0.0;
    double uy = 0.0;

    for (int i = 0; i < 9; ++i) {
        // Indexación SoA perfectamente coalescente para la GPU
        double fi = f[i * NX * NY + idx];
        rho += fi;
        ux  += fi * d_cx[i];
        uy  += fi * d_cy[i];
    }
    ux /= rho;
    uy /= rho;

    // Aplicar relajación BGK
    double u2 = ux * ux + uy * uy;
    for (int i = 0; i < 9; ++i) {
        double cu = d_cx[i] * ux + d_cy[i] * uy;
        double feq = d_w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        int f_idx = i * NX * NY + idx;
        f[f_idx] = f[f_idx] - (f[f_idx] - feq) / tau_param;
    }
}

// KERNEL 2: Streaming de tipo "Pull" en GPU
__global__ void streaming_kernel(const double* f, double* f_next, const bool* obstacle, double u_inflow_param) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= NY) return;
    int idx = y * NX + x;

    // CONDICIÓN DE FRONTERA: Entrada (Izquierda)
    if (x == 0) {
        double u2 = u_inflow_param * u_inflow_param;
        for (int i = 0; i < 9; ++i) {
            double cu = d_cx[i] * u_inflow_param;
            f_next[i * NX * NY + idx] = d_w[i] * 1.0 * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        }
        return;
    }

    // CONDICIÓN DE FRONTERA: Salida (Derecha) libre de carreras (Race-free Pull)
    if (x == NX - 1) {
        int virtual_x = NX - 2;
        int v_idx = y * NX + virtual_x;
        for (int i = 0; i < 9; ++i) {
            int prev_x = virtual_x - d_cx[i];
            int prev_y = y - d_cy[i];
            if (prev_y < 0 || prev_y >= NY) {
                f_next[i * NX * NY + idx] = f[d_noslip[i] * NX * NY + v_idx];
            } else if (obstacle[prev_y * NX + prev_x]) {
                f_next[i * NX * NY + idx] = f[d_noslip[i] * NX * NY + v_idx];
            } else {
                f_next[i * NX * NY + idx] = f[i * NX * NY + (prev_y * NX + prev_x)];
            }
        }
        return;
    }

    if (obstacle[idx]) return;

    // STREAMING NORMAL INTERIOR
    for (int i = 0; i < 9; ++i) {
        int prev_x = x - d_cx[i];
        int prev_y = y - d_cy[i];
        int prev_idx = prev_y * NX + prev_x;

        if (prev_y < 0 || prev_y >= NY) {
            f_next[i * NX * NY + idx] = f[d_noslip[i] * NX * NY + idx];
        } else if (obstacle[prev_idx]) {
            f_next[i * NX * NY + idx] = f[d_noslip[i] * NX * NY + idx];
        } else {
            f_next[i * NX * NY + idx] = f[i * NX * NY + prev_idx];
        }
    }
}

// Función de guardado VTK (Adaptada a la lectura del formato SoA)
void save_vtk(int step, const std::vector<double>& h_f, const std::vector<bool>& h_obstacle) {
    namespace fs = std::filesystem;
    fs::path out_dir = fs::path(std::getenv("HOME")) / "Documents/Proyectos/Informática/HPC/LMB_Autotuning/results/phase3";
    fs::create_directories(out_dir);

    std::stringstream ss;
    ss << "fluid_" << std::setw(4) << std::setfill('0') << step << ".vtk";
    std::ofstream out((out_dir / ss.str()).string());

    out << "# vtk DataFile Version 3.0\nLBM 2D9Q GPU\nASCII\nDATASET STRUCTURED_POINTS\n";
    out << "DIMENSIONS " << NX << " " << NY << " 1\nORIGIN 0 0 0\nSPACING 1 1 1\n";
    out << "POINT_DATA " << NX * NY << "\nSCALARS velocity_magnitude double 1\nLOOKUP_TABLE default\n";

    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = y * NX + x;
            if (h_obstacle[idx]) {
                out << 0.0 << "\n";
            } else {
                double rho = 0, ux = 0, uy = 0;
                for (int i = 0; i < 9; ++i) {
                    double fi = h_f[i * NX * NY + idx];
                    rho += fi;
                    ux  += fi * cx[i];
                    uy  += fi * cy[i];
                }
                out << std::sqrt((ux/rho)*(ux/rho) + (uy/rho)*(uy/rho)) << "\n";
            }
        }
    }

    out << "VECTORS velocity double\n";
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = y * NX + x;
            if (h_obstacle[idx]) {
                out << "0.0 0.0 0.0\n";
            } else {
                double rho = 0, ux = 0, uy = 0;
                for (int i = 0; i < 9; ++i) {
                    double fi = h_f[i * NX * NY + idx];
                    rho += fi;
                    ux  += fi * cx[i];
                    uy  += fi * cy[i];
                }
                out << ux/rho << " " << uy/rho << " 0.0\n";
            }
        }
    }
}

int main() {
    // 1. Reservar memoria en el HOST (CPU) usando formato SoA plano
    std::vector<double> h_f(9 * NX * NY);
    std::vector<bool> h_obstacle(NX * NY, false);

    // Inicializar obstáculo
    int cx_cyl = NX / 4, cy_cyl = NY / 2, r_cyl = NY / 10;
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            if ((x - cx_cyl)*(x - cx_cyl) + (y - cy_cyl)*(y - cy_cyl) < r_cyl * r_cyl) {
                h_obstacle[y * NX + x] = true;
            }
        }
    }

    // Inicializar densidades en equilibrio (SoA)
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = y * NX + x;
            // Ecuación de equilibrio inicial
            double cu_init = cx[0] * u_inflow; // Simplificado para t=0
            double u2_init = u_inflow * u_inflow;
            for (int i = 0; i < 9; ++i) {
                double cu = cx[i] * u_inflow;
                h_f[i * NX * NY + idx] = w[i] * 1.0 * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2_init);
            }
        }
    }

    // Necesitamos pasar el vector de bool a un formato plano compatible con C/CUDA
    std::vector<char> h_obs_char(NX * NY);
    for(int i=0; i<NX*NY; ++i) h_obs_char[i] = h_obstacle[i] ? 1 : 0;

    // 2. Reservar memoria en el DEVICE (GPU)
    double *d_f, *d_f_next;
    bool *d_obstacle;
    cudaMalloc(&d_f, 9 * NX * NY * sizeof(double));
    cudaMalloc(&d_f_next, 9 * NX * NY * sizeof(double));
    cudaMalloc(&d_obstacle, NX * NY * sizeof(bool));

    // 3. Copiar datos iniciales de CPU a GPU
    cudaMemcpy(d_f, h_f.data(), 9 * NX * NY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obstacle, h_obs_char.data(), NX * NY * sizeof(bool), cudaMemcpyHostToDevice);

    // Copiar vectores geométricos a la Memoria Constante de la GPU
    cudaMemcpyToSymbol(d_cx, cx, 9 * sizeof(int));
    cudaMemcpyToSymbol(d_cy, cy, 9 * sizeof(int));
    cudaMemcpyToSymbol(d_w, w, 9 * sizeof(double));
    cudaMemcpyToSymbol(d_noslip, noslip, 9 * sizeof(int));

    // 4. Configurar la topología de hilos de la GPU (Bloques y Rejilla)
    dim3 blockSize(32, 8); // Bloques de 256 hilos (32x8)
    dim3 gridSize((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y);

    std::cout << "Iniciando simulación en GPU (CUDA)..." << std::endl;

    // Eventos de CUDA para medir el tiempo exacto dentro de la tarjeta gráfica
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // --- BUCLE PRINCIPAL EN GPU ---
    for (int step = 0; step <= NUM_STEPS; ++step) {
        
        // Lanzar Kernel 1: Colisión
        collision_kernel<<<gridSize, blockSize>>>(d_f, d_obstacle, tau);

        // Lanzar Kernel 2: Streaming (Pull)
        streaming_kernel<<<gridSize, blockSize>>>(d_f, d_f_next, d_obstacle, u_inflow);

        // Intercambio de punteros en la GPU (Pointer Swap instantáneo)
        std::swap(d_f, d_f_next);

        // Cada X pasos bajamos los datos para guardar el VTK
        if (step % SAVE_EVERY == 0) {
            std::cout << "Paso GPU: " << step << " / " << NUM_STEPS << std::endl;
            // Traer los datos calculados de la VRAM a la RAM
            cudaMemcpy(h_f.data(), d_f, 9 * NX * NY * sizeof(double), cudaMemcpyDeviceToHost);
            save_vtk(step, h_f, h_obstacle);
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "=========================================================" << std::endl;
    std::cout << "¡Simulación en GPU completada!" << std::endl;
    std::cout << "Tiempo de cómputo en tarjeta gráfica: " << (milliseconds / 1000.0) << " segundos." << std::endl;
    std::cout << "=========================================================" << std::endl;

    // Liberar memoria de la GPU
    cudaFree(d_f);
    cudaFree(d_f_next);
    cudaFree(d_obstacle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}