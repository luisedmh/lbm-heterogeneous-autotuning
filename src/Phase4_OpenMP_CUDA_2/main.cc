#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstring>
#include <omp.h>
#include <cuda_runtime.h>

// Dimensiones de la simulación
const int NX = 1000;
const int NY = 1000;
const int NUM_STEPS = 3000;

// Parámetros teóricos de partición
const double ALPHA = 0.8235; 
const int NY_GPU = static_cast<int>(NY * ALPHA); // 823 filas
const int NY_CPU = NY - NY_GPU;                  // 177 filas
const int NY_INTERIOR = NY_GPU - 1;              // 822 filas interiores GPU

// Parámetros físicos
const double tau = 0.56; 
const double u_inflow = 0.1;

// Vectores discretos D2Q9
const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
const double w[9] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 
                     1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
const int noslip[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6}; 

// Memoria Constante GPU
__constant__ int d_cx[9];
__constant__ int d_cy[9];
__constant__ double d_w[9];
__constant__ int d_noslip[9];

// ============================================================================
// KERNELS DE GPU
// ============================================================================

// KERNEL 1: Colisión local en todo el subdominio GPU
__global__ void collision_kernel_gpu(double* f, const bool* obstacle, double tau_param, int ny_sub) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= ny_sub) return;
    int idx = y * NX + x;   // índice de la casilla actual

    if (obstacle[idx]) return;

    double rho = 0.0, ux = 0.0, uy = 0.0;
    for (int i = 0; i < 9; ++i) {   // Para cada dirección de velocidad
        double fi = f[i * NX * ny_sub + idx];
        rho += fi;
        ux  += fi * d_cx[i];
        uy  += fi * d_cy[i];
    }
    ux /= rho;
    uy /= rho;

    double u2 = ux * ux + uy * uy;
    for (int i = 0; i < 9; ++i) {   // Para cada dirección de velocidad
        double cu = d_cx[i] * ux + d_cy[i] * uy;
        double feq = d_w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        int f_idx = i * NX * ny_sub + idx;
        f[f_idx] = f[f_idx] - (f[f_idx] - feq) / tau_param;
    }
}

// KERNEL 2: Streaming exclusivo de la Fila Frontera (y = ny_sub - 1)
// Empaqueta directamente en 'halo_out' para emitir 1 sola copia PCIe contigua
__global__ void streaming_boundary_kernel(const double* f, double* f_next, double* halo_out, 
                                          const bool* obstacle, double u_inflow_param, int ny_sub) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // Solo una fila
    if (x >= NX) return;

    int y = ny_sub - 1; // Última fila de la GPU
    int idx = y * NX + x;   // índice de la casilla actual

    if (x == 0) {
        double u2 = u_inflow_param * u_inflow_param;
        for (int i = 0; i < 9; ++i) {   // Para cada dirección de velocidad
            double cu = d_cx[i] * u_inflow_param;
            double val = d_w[i] * 1.0 * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
            f_next[i * NX * ny_sub + idx] = val;    // Actualiza la frontera de la GPU
            halo_out[i * NX + x] = val; // Actualiza el halo contiguo para PCIe
        }
        return;
    }

    if (x == NX - 1) {
        int virtual_x = NX - 2;
        int v_idx = y * NX + virtual_x;
        for (int i = 0; i < 9; ++i) {   // Para cada dirección de velocidad
            int prev_x = virtual_x - d_cx[i];
            int prev_y = y - d_cy[i];
            double val;
            if (prev_y < 0 || prev_y >= ny_sub || obstacle[prev_y * NX + prev_x]) {
                val = f[d_noslip[i] * NX * ny_sub + v_idx];
            } else {
                val = f[i * NX * ny_sub + (prev_y * NX + prev_x)];
            }
            f_next[i * NX * ny_sub + idx] = val;    // Actualiza la frontera de la GPU
            halo_out[i * NX + x] = val; // Actualiza el halo contiguo para PCIe<
        }
        return;
    }

    if (obstacle[idx]) return;

    for (int i = 0; i < 9; ++i) {   // Para cada dirección de velocidad
        int prev_x = x - d_cx[i];
        int prev_y = y - d_cy[i];
        int prev_idx = prev_y * NX + prev_x;
        double val;
        if (prev_y < 0 || prev_y >= ny_sub || obstacle[prev_idx]) {
            val = f[d_noslip[i] * NX * ny_sub + idx];
        } else {
            val = f[i * NX * ny_sub + prev_idx];
        }
        f_next[i * NX * ny_sub + idx] = val;    // Actualiza la frontera de la GPU
        halo_out[i * NX + x] = val; // Actualiza el halo contiguo para PCIe
    }
}

// KERNEL 3: Streaming de la masa interior de la GPU (y = 0 hasta ny_interior - 1)
__global__ void streaming_interior_kernel(const double* f, double* f_next, const bool* obstacle, 
                                          double u_inflow_param, int ny_interior, int ny_sub_total) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= NX || y >= ny_interior) return;
    int idx = y * NX + x;

    if (x == 0) {
        double u2 = u_inflow_param * u_inflow_param;
        for (int i = 0; i < 9; ++i) {
            double cu = d_cx[i] * u_inflow_param;
            f_next[i * NX * ny_sub_total + idx] = d_w[i] * 1.0 * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        }
        return;
    }

    if (x == NX - 1) {
        int virtual_x = NX - 2;
        int v_idx = y * NX + virtual_x;
        for (int i = 0; i < 9; ++i) {
            int prev_x = virtual_x - d_cx[i];
            int prev_y = y - d_cy[i];
            if (prev_y < 0 || prev_y >= ny_sub_total || obstacle[prev_y * NX + prev_x]) {
                f_next[i * NX * ny_sub_total + idx] = f[d_noslip[i] * NX * ny_sub_total + v_idx];
            } else {
                f_next[i * NX * ny_sub_total + idx] = f[i * NX * ny_sub_total + (prev_y * NX + prev_x)];
            }
        }
        return;
    }

    if (obstacle[idx]) return;

    for (int i = 0; i < 9; ++i) {
        int prev_x = x - d_cx[i];
        int prev_y = y - d_cy[i];
        int prev_idx = prev_y * NX + prev_x;

        if (prev_y < 0 || prev_y >= ny_sub_total || obstacle[prev_idx]) {
            f_next[i * NX * ny_sub_total + idx] = f[d_noslip[i] * NX * ny_sub_total + idx];
        } else {
            f_next[i * NX * ny_sub_total + idx] = f[i * NX * ny_sub_total + prev_idx];
        }
    }
}

// ============================================================================
// CÓDIGO CPU (OPENMP)
// ============================================================================
void cpu_lbm_step(std::vector<double>& f, std::vector<double>& f_next, const std::vector<char>& obs) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = NY_GPU; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = y * NX + x;
            if (obs[idx]) continue;

            // Colisión
            double rho = 0.0, ux = 0.0, uy = 0.0;
            for (int i = 0; i < 9; ++i) {
                double fi = f[i * NX * NY + idx];
                rho += fi;
                ux  += fi * cx[i];
                uy  += fi * cy[i];
            }
            ux /= rho; uy /= rho;
            double u2 = ux * ux + uy * uy;

            for (int i = 0; i < 9; ++i) {
                double cu = cx[i] * ux + cy[i] * uy;
                double feq = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
                int f_idx = i * NX * NY + idx;
                f[f_idx] = f[f_idx] - (f[f_idx] - feq) / tau;
            }

            // Streaming
            if (x == 0) {
                double u2_in = u_inflow * u_inflow;
                for (int i = 0; i < 9; ++i) {
                    double cu = cx[i] * u_inflow;
                    f_next[i * NX * NY + idx] = w[i] * 1.0 * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2_in);
                }
            } else {
                for (int i = 0; i < 9; ++i) {
                    int prev_x = x - cx[i];
                    int prev_y = y - cy[i];
                    int prev_idx = prev_y * NX + prev_x;

                    if (prev_y < 0 || prev_y >= NY || obs[prev_idx]) {
                        f_next[i * NX * NY + idx] = f[noslip[i] * NX * NY + idx];
                    } else {
                        f_next[i * NX * NY + idx] = f[i * NX * NY + prev_idx];
                    }
                }
            }
        }
    }
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    std::cout << "=========================================================" << std::endl;
    std::cout << ">>> FASE 4: CO-EJECUCIÓN TOTALMENTE SOLAPADA (GPU+PCIe+CPU)" << std::endl;
    std::cout << "Reparto -> GPU: " << ALPHA*100 << "% (" << NY_GPU << " filas) | CPU: " 
              << (1-ALPHA)*100 << "% (" << NY_CPU << " filas)" << std::endl;
    std::cout << "=========================================================" << std::endl;

    // 1. Memoria Host
    std::vector<double> h_f(9 * NX * NY);
    std::vector<double> h_f_next(9 * NX * NY);
    std::vector<char> h_obs(NX * NY, 0);

    // Inicializar obstáculo
    int cx_cyl = NX / 4, cy_cyl = NY / 2, r_cyl = NY / 10;
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            if ((x - cx_cyl)*(x - cx_cyl) + (y - cy_cyl)*(y - cy_cyl) < r_cyl * r_cyl) {
                h_obs[y * NX + x] = 1;
            }
        }
    }

    // Inicializar densidades en equilibrio
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

    // 2. Memoria Device (GPU)
    double *d_f, *d_f_next;
    bool *d_obstacle;
    cudaMalloc(&d_f, 9 * NX * NY_GPU * sizeof(double));
    cudaMalloc(&d_f_next, 9 * NX * NY_GPU * sizeof(double));
    cudaMalloc(&d_obstacle, NX * NY_GPU * sizeof(bool));

    std::vector<char> h_obs_bool_gpu(NX * NY_GPU);
    for(int i = 0; i < NX * NY_GPU; ++i) h_obs_bool_gpu[i] = (h_obs[i] == 1);

    for (int i = 0; i < 9; ++i) {
        cudaMemcpy(d_f + i * NX * NY_GPU, h_f.data() + i * NX * NY, NX * NY_GPU * sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_obstacle, h_obs_bool_gpu.data(), NX * NY_GPU * sizeof(bool), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_cx, cx, 9 * sizeof(int));
    cudaMemcpyToSymbol(d_cy, cy, 9 * sizeof(int));
    cudaMemcpyToSymbol(d_w, w, 9 * sizeof(double));
    cudaMemcpyToSymbol(d_noslip, noslip, 9 * sizeof(int));

    // 3. Búferes del Halo en Memoria Pinned (DMA real no bloqueante)
    double *h_halo_pinned;
    double *d_halo_gpu;
    size_t halo_size_bytes = 9 * NX * sizeof(double);
    cudaMallocHost(&h_halo_pinned, halo_size_bytes); // Memoria fijada en Host para no depender del SO
    cudaMalloc(&d_halo_gpu, halo_size_bytes);        // Memoria contigua en VRAM

    // 4. Configuración de Rejillas y Streams
    dim3 blockSize(32, 8);
    dim3 gridCollision((NX + blockSize.x - 1) / blockSize.x, (NY_GPU + blockSize.y - 1) / blockSize.y);
    dim3 gridInterior((NX + blockSize.x - 1) / blockSize.x, (NY_INTERIOR + blockSize.y - 1) / blockSize.y);

    dim3 blockBoundary(256);
    dim3 gridBoundary((NX + blockBoundary.x - 1) / blockBoundary.x);

    cudaStream_t stream_halo, stream_interior;
    cudaStreamCreate(&stream_halo);
    cudaStreamCreate(&stream_interior);

    cudaEvent_t event_collision_done;
    cudaEventCreate(&event_collision_done);

    auto start_time = std::chrono::high_resolution_clock::now();

    // ========================================================================
    // BUCLE PRINCIPAL DE CO-EJECUCIÓN ASÍNCRONA
    // ========================================================================
    for (int step = 0; step < NUM_STEPS; ++step) {

        // A. La GPU arranca la colisión completa en stream_halo
        collision_kernel_gpu<<<gridCollision, blockSize, 0, stream_halo>>>(d_f, d_obstacle, tau, NY_GPU);
        
        // Registra el evento en hardware de que la colisión terminó
        cudaEventRecord(event_collision_done, stream_halo);

        // B. Stream Halo: calcula la Fila Frontera (822) e inmediatamente dispara el DMA por PCIe
        streaming_boundary_kernel<<<gridBoundary, blockBoundary, 0, stream_halo>>>(
            d_f, d_f_next, d_halo_gpu, d_obstacle, u_inflow, NY_GPU
        );
        cudaMemcpyAsync(h_halo_pinned, d_halo_gpu, halo_size_bytes, cudaMemcpyDeviceToHost, stream_halo);

        // C. Stream Interior: espera a que termine la colisión sin bloquear a la CPU
        cudaStreamWaitEvent(stream_interior, event_collision_done, 0);
        streaming_interior_kernel<<<gridInterior, blockSize, 0, stream_interior>>>(
            d_f, d_f_next, d_obstacle, u_inflow, NY_INTERIOR, NY_GPU
        );

        // D. La CPU ejecuta su cálculo OpenMP (concurrente con la GPU y el DMA)
        cpu_lbm_step(h_f, h_f_next, h_obs);

        // E. Sincronización al final del paso t
        cudaStreamSynchronize(stream_halo);
        cudaStreamSynchronize(stream_interior);

        // F. Desempaquetar el halo recibido a la frontera de la CPU (en memoria L1/L2)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < 9; ++i) {
            std::memcpy(h_f_next.data() + i * NX * NY + NY_GPU * NX,
                        h_halo_pinned + i * NX,
                        NX * sizeof(double));
        }

        // Intercambio de punteros
        std::swap(d_f, d_f_next);
        h_f.swap(h_f_next);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(end_time - start_time).count();
    double mlups = (double(NX) * NY * NUM_STEPS) / (total_sec * 1e6);

    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "Tiempo Total HETEROGÉNEO (CPU+GPU+PCIe): " << total_sec << " segundos." << std::endl;
    std::cout << "Rendimiento Heterogéneo               : " << mlups << " MLUPS" << std::endl;
    std::cout << "=========================================================" << std::endl;

    // Liberación de recursos
    cudaFreeHost(h_halo_pinned);
    cudaFree(d_halo_gpu);
    cudaFree(d_f);
    cudaFree(d_f_next);
    cudaFree(d_obstacle);
    cudaStreamDestroy(stream_halo);
    cudaStreamDestroy(stream_interior);
    cudaEventDestroy(event_collision_done);

    return 0;
}