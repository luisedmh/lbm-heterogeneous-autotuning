// ============================================================================
// LBM D2Q9 - HETEROGENEO GPU+CPU con REBALANCEO DINAMICO (fp64 por defecto).
//
// Parte de lbm_heterogeneo_v2.cu (reparto fijo, ya validado a nivel de
// maquina epsilon) y le anade la pieza que faltaba: en vez de fijar ny_gpu
// una vez al principio, el programa MIDE en tiempo real cuanto tarda cada
// lado por paso y mueve la frontera ny_gpu hacia el punto de equilibrio,
// sin que tu le digas de antemano cual es el reparto correcto.
//
// Los kernels de GPU y las funciones de CPU son EXACTAMENTE los mismos que
// en lbm_heterogeneo_v2.cu (no cambia la fisica ni el patron
// bootstrap/fusionado/unwrap). Lo nuevo esta todo en main():
//
//   1) Cronometrar cada lado por separado cada paso:
//        - GPU: con cudaEvent (mide SOLO el tiempo del kernel, en el propio
//          stream, no el tiempo de pared del host).
//        - CPU: con std::chrono (codigo de host, sincrono, se mide solo).
//
//   2) Cada REBALANCE_WINDOW pasos, con el tiempo medio acumulado de cada
//      lado, recalcular el reparto que igualaria los dos tiempos (la misma
//      formula que usamos a mano para predecir ny_gpu=724 en fp64, pero
//      ahora el programa la recalcula solo, con datos frescos, sin que
//      nadie le diga los microsegundos por fila de antemano).
//
//   3) Si el reparto ideal se aleja del actual, MOVER la frontera: como
//      tanto d_f (GPU) como h_f (CPU) reservan el dominio COMPLETO desde el
//      principio (por diseno, ver el comentario de NY_GPU_INIT mas abajo),
//      mover la frontera es simplemente copiar las filas que cambian de
//      dueno de un lado al otro (cudaMemcpy2D, porque el array guarda las 9
//      direcciones en planos separados, no fila a fila) y volver a calcular
//      cual es la fila de contacto para los dos halos.
//
// Un cambio de reparto se limita (MAX_STEP_ROWS) para no dar saltos bruscos
// por una medida ruidosa, y se acota (NY_GPU_MIN/MAX) para no acercarse al
// caso ny_gpu=0 (que tiene un caso limite conocido en la condicion de
// frontera, documentado en lbm_heterogeneo_v2.cu, y que no hace falta tocar
// aqui porque en fp64 el equilibrio esta lejos de los extremos).
//
// Cada vez que se mueve la frontera se anade una linea a
// rebalanceo_log.csv (paso, ny_gpu, us/fila de cada lado) para poder
// dibujar despues como converge el reparto a lo largo de la simulacion.
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>

const int NX = 1000;
const int NY = 1000;
const int NUM_STEPS = 3000;

// Reparto INICIAL: a proposito lejos del optimo (50/50) para que se vea de
// verdad la convergencia en rebalanceo_log.csv, en vez de "hacer trampa"
// empezando ya cerca de 724.
const int NY_GPU_INIT = 500;

// Cada cuantos pasos se recalcula el reparto. Mas pequeno = reacciona antes
// pero paga mas veces el coste de migrar filas y es mas sensible al ruido
// de medir un solo paso; mas grande = mas estable pero tarda mas en llegar
// al equilibrio.
const int REBALANCE_WINDOW = 30;

// Maximo de filas que se mueven de golpe en un solo reequilibrio (amortigua
// saltos grandes si una medida sale ruidosa).
const int MAX_STEP_ROWS = 120;

// No dejar que ningun lado se quede con menos de esto (evita acercarse al
// caso limite ny_gpu=0 / ny_gpu=NY y deja margen para medir un tiempo
// minimamente estable).
const int NY_GPU_MIN = 20;
const int NY_GPU_MAX = NY - 20;

#ifdef USE_SINGLE_PRECISION
    using real_t = float;
    #define REAL_LIT(x) x##f
#else
    using real_t = double;
    #define REAL_LIT(x) x
#endif

const real_t tau = REAL_LIT(0.56);
const real_t u_inflow = REAL_LIT(0.1);

const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
const real_t w[9] = {REAL_LIT(4.0)/REAL_LIT(9.0), REAL_LIT(1.0)/REAL_LIT(9.0), REAL_LIT(1.0)/REAL_LIT(9.0),
                      REAL_LIT(1.0)/REAL_LIT(9.0), REAL_LIT(1.0)/REAL_LIT(9.0),
                      REAL_LIT(1.0)/REAL_LIT(36.0), REAL_LIT(1.0)/REAL_LIT(36.0),
                      REAL_LIT(1.0)/REAL_LIT(36.0), REAL_LIT(1.0)/REAL_LIT(36.0)};
const int noslip[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

__constant__ int d_cx[9];
__constant__ int d_cy[9];
__constant__ real_t d_w[9];
__constant__ int d_noslip[9];

// ============================================================================
// GPU: gather de una direccion en (x,y), consciente de la pared superior
// (y=0) y de la frontera con la CPU (ny_gpu). Identico a lbm_heterogeneo_v2.cu.
// ============================================================================
__device__ __forceinline__ real_t gpu_gather_dir(
    const real_t* __restrict__ f, const bool* __restrict__ obstacle_full,
    const real_t* __restrict__ halo_from_cpu,
    int x, int y, int i, int ny_gpu, int N)
{
    int prev_x = x - d_cx[i];
    int prev_y = y - d_cy[i];
    int idx = y * NX + x;
    if (prev_y < 0) {
        return f[d_noslip[i] * N + idx];
    } else if (prev_y >= ny_gpu) {
        if (obstacle_full[prev_y * NX + prev_x]) return f[d_noslip[i] * N + idx];
        else return halo_from_cpu[i * NX + prev_x];
    } else if (obstacle_full[prev_y * NX + prev_x]) {
        return f[d_noslip[i] * N + idx];
    } else {
        return f[i * N + (prev_y * NX + prev_x)];
    }
}

__global__ void collision_only_het(real_t* __restrict__ f, const bool* __restrict__ obstacle_full,
                                    real_t* __restrict__ halo_to_cpu_out, real_t tau_param, int ny_gpu) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= ny_gpu) return;
    int idx = y * NX + x;
    const int N = NX * NY;
    if (obstacle_full[idx]) return;

    real_t rho = REAL_LIT(0.0), ux = REAL_LIT(0.0), uy = REAL_LIT(0.0);
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        real_t fi = f[i * N + idx];
        rho += fi; ux += fi * d_cx[i]; uy += fi * d_cy[i];
    }
    ux /= rho; uy /= rho;
    real_t u2 = ux * ux + uy * uy;
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        real_t cu = d_cx[i] * ux + d_cy[i] * uy;
        real_t feq = d_w[i] * rho * (REAL_LIT(1.0) + REAL_LIT(3.0) * cu + REAL_LIT(4.5) * cu * cu - REAL_LIT(1.5) * u2);
        int f_idx = i * N + idx;
        f[f_idx] = f[f_idx] - (f[f_idx] - feq) / tau_param;
    }
    if (y == ny_gpu - 1) {
        #pragma unroll
        for (int i = 0; i < 9; ++i) halo_to_cpu_out[i * NX + x] = f[i * N + idx];
    }
}

__global__ void streaming_only_het(const real_t* __restrict__ f, real_t* __restrict__ f_next,
                                    const bool* __restrict__ obstacle_full,
                                    const real_t* __restrict__ halo_from_cpu,
                                    real_t u_inflow_param, int ny_gpu) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= ny_gpu) return;
    int idx = y * NX + x;
    const int N = NX * NY;

    if (x == 0) {
        real_t u2 = u_inflow_param * u_inflow_param;
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            real_t cu = d_cx[i] * u_inflow_param;
            f_next[i * N + idx] = d_w[i] * (REAL_LIT(1.0) + REAL_LIT(3.0) * cu + REAL_LIT(4.5) * cu * cu - REAL_LIT(1.5) * u2);
        }
        return;
    }
    if (x == NX - 1) {
        int virtual_x = NX - 2;
        int v_idx = y * NX + virtual_x;
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            int prev_x = virtual_x - d_cx[i];
            int prev_y = y - d_cy[i];
            real_t val;
            if (prev_y < 0) {
                val = f[d_noslip[i] * N + v_idx];
            } else if (prev_y >= ny_gpu) {
                val = obstacle_full[prev_y * NX + prev_x] ? f[d_noslip[i] * N + v_idx]
                                                           : halo_from_cpu[i * NX + prev_x];
            } else if (obstacle_full[prev_y * NX + prev_x]) {
                val = f[d_noslip[i] * N + v_idx];
            } else {
                val = f[i * N + (prev_y * NX + prev_x)];
            }
            f_next[i * N + idx] = val;
        }
        return;
    }
    if (obstacle_full[idx]) return;
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        f_next[i * N + idx] = gpu_gather_dir(f, obstacle_full, halo_from_cpu, x, y, i, ny_gpu, N);
    }
}

__global__ void collide_stream_fused_het(const real_t* __restrict__ f, real_t* __restrict__ f_next,
                                          const bool* __restrict__ obstacle_full,
                                          const real_t* __restrict__ halo_from_cpu,
                                          real_t* __restrict__ halo_to_cpu_out,
                                          real_t tau_param, real_t u_inflow_param, int ny_gpu) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= ny_gpu) return;
    int idx = y * NX + x;
    const int N = NX * NY;

    if (x == 0) {
        real_t u2 = u_inflow_param * u_inflow_param;
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            real_t cu = d_cx[i] * u_inflow_param;
            f_next[i * N + idx] = d_w[i] * (REAL_LIT(1.0) + REAL_LIT(3.0) * cu + REAL_LIT(4.5) * cu * cu - REAL_LIT(1.5) * u2);
        }
        if (y == ny_gpu - 1) {
            #pragma unroll
            for (int i = 0; i < 9; ++i) halo_to_cpu_out[i * NX + x] = f_next[i * N + idx];
        }
        return;
    }
    if (x == NX - 1) {
        int virtual_x = NX - 2;
        real_t g[9];
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            int prev_x = virtual_x - d_cx[i];
            int prev_y = y - d_cy[i];
            int v_idx = y * NX + virtual_x;
            if (prev_y < 0) {
                g[i] = f[d_noslip[i] * N + v_idx];
            } else if (prev_y >= ny_gpu) {
                g[i] = obstacle_full[prev_y * NX + prev_x] ? f[d_noslip[i] * N + v_idx]
                                                            : halo_from_cpu[i * NX + prev_x];
            } else if (obstacle_full[prev_y * NX + prev_x]) {
                g[i] = f[d_noslip[i] * N + v_idx];
            } else {
                g[i] = f[i * N + (prev_y * NX + prev_x)];
            }
        }
        real_t rho = REAL_LIT(0.0), ux = REAL_LIT(0.0), uy = REAL_LIT(0.0);
        #pragma unroll
        for (int i = 0; i < 9; ++i) { rho += g[i]; ux += g[i] * d_cx[i]; uy += g[i] * d_cy[i]; }
        ux /= rho; uy /= rho;
        real_t u2 = ux * ux + uy * uy;
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            real_t cu = d_cx[i] * ux + d_cy[i] * uy;
            real_t feq = d_w[i] * rho * (REAL_LIT(1.0) + REAL_LIT(3.0) * cu + REAL_LIT(4.5) * cu * cu - REAL_LIT(1.5) * u2);
            f_next[i * N + idx] = g[i] - (g[i] - feq) / tau_param;
        }
        if (y == ny_gpu - 1) {
            #pragma unroll
            for (int i = 0; i < 9; ++i) halo_to_cpu_out[i * NX + x] = f_next[i * N + idx];
        }
        return;
    }
    if (obstacle_full[idx]) return;

    real_t g[9];
    #pragma unroll
    for (int i = 0; i < 9; ++i) g[i] = gpu_gather_dir(f, obstacle_full, halo_from_cpu, x, y, i, ny_gpu, N);
    real_t rho = REAL_LIT(0.0), ux = REAL_LIT(0.0), uy = REAL_LIT(0.0);
    #pragma unroll
    for (int i = 0; i < 9; ++i) { rho += g[i]; ux += g[i] * d_cx[i]; uy += g[i] * d_cy[i]; }
    ux /= rho; uy /= rho;
    real_t u2 = ux * ux + uy * uy;
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        real_t cu = d_cx[i] * ux + d_cy[i] * uy;
        real_t feq = d_w[i] * rho * (REAL_LIT(1.0) + REAL_LIT(3.0) * cu + REAL_LIT(4.5) * cu * cu - REAL_LIT(1.5) * u2);
        f_next[i * N + idx] = g[i] - (g[i] - feq) / tau_param;
    }
    if (y == ny_gpu - 1) {
        #pragma unroll
        for (int i = 0; i < 9; ++i) halo_to_cpu_out[i * NX + x] = f_next[i * N + idx];
    }
}

// ============================================================================
// CPU: identico a lbm_heterogeneo_v2.cu.
// ============================================================================
static inline real_t cpu_gather_dir(const std::vector<real_t>& f, const std::vector<char>& obs,
                                     const std::vector<real_t>& halo_from_gpu,
                                     int x, int y, int i, int ny_gpu) {
    int prev_x = x - cx[i];
    int prev_y = y - cy[i];
    int idx = y * NX + x;
    if (prev_y >= NY) {
        return f[noslip[i] * NX * NY + idx];
    } else if (prev_y < ny_gpu) {
        if (obs[prev_y * NX + prev_x]) return f[noslip[i] * NX * NY + idx];
        else return halo_from_gpu[i * NX + prev_x];
    } else if (obs[prev_y * NX + prev_x]) {
        return f[noslip[i] * NX * NY + idx];
    } else {
        return f[i * NX * NY + (prev_y * NX + prev_x)];
    }
}

void cpu_collision_only(std::vector<real_t>& f, const std::vector<char>& obs,
                         std::vector<real_t>& halo_to_gpu_out, int ny_gpu) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = ny_gpu; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = y * NX + x;
            if (obs[idx]) continue;
            real_t rho = REAL_LIT(0.0), ux = REAL_LIT(0.0), uy = REAL_LIT(0.0);
            for (int i = 0; i < 9; ++i) {
                real_t fi = f[i * NX * NY + idx];
                rho += fi; ux += fi * cx[i]; uy += fi * cy[i];
            }
            ux /= rho; uy /= rho;
            real_t u2 = ux * ux + uy * uy;
            for (int i = 0; i < 9; ++i) {
                real_t cu = cx[i] * ux + cy[i] * uy;
                real_t feq = w[i] * rho * (REAL_LIT(1.0) + REAL_LIT(3.0) * cu + REAL_LIT(4.5) * cu * cu - REAL_LIT(1.5) * u2);
                int f_idx = i * NX * NY + idx;
                f[f_idx] = f[f_idx] - (f[f_idx] - feq) / tau;
            }
            if (y == ny_gpu) {
                for (int i = 0; i < 9; ++i) halo_to_gpu_out[i * NX + x] = f[i * NX * NY + idx];
            }
        }
    }
}

void cpu_streaming_only(const std::vector<real_t>& f, std::vector<real_t>& f_next,
                         const std::vector<char>& obs, const std::vector<real_t>& halo_from_gpu, int ny_gpu) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = ny_gpu; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = y * NX + x;
            if (x == 0) {
                real_t u2 = u_inflow * u_inflow;
                for (int i = 0; i < 9; ++i) {
                    real_t cu = cx[i] * u_inflow;
                    f_next[i * NX * NY + idx] = w[i] * (REAL_LIT(1.0) + REAL_LIT(3.0) * cu + REAL_LIT(4.5) * cu * cu - REAL_LIT(1.5) * u2);
                }
                continue;
            }
            if (x == NX - 1) {
                int virtual_x = NX - 2;
                int v_idx = y * NX + virtual_x;
                for (int i = 0; i < 9; ++i) {
                    int prev_x = virtual_x - cx[i];
                    int prev_y = y - cy[i];
                    real_t val;
                    if (prev_y >= NY) {
                        val = f[noslip[i] * NX * NY + v_idx];
                    } else if (prev_y < ny_gpu) {
                        val = obs[prev_y * NX + prev_x] ? f[noslip[i] * NX * NY + v_idx]
                                                         : halo_from_gpu[i * NX + prev_x];
                    } else if (obs[prev_y * NX + prev_x]) {
                        val = f[noslip[i] * NX * NY + v_idx];
                    } else {
                        val = f[i * NX * NY + (prev_y * NX + prev_x)];
                    }
                    f_next[i * NX * NY + idx] = val;
                }
                continue;
            }
            if (obs[idx]) continue;
            for (int i = 0; i < 9; ++i)
                f_next[i * NX * NY + idx] = cpu_gather_dir(f, obs, halo_from_gpu, x, y, i, ny_gpu);
        }
    }
}

void cpu_fused_step(const std::vector<real_t>& f, std::vector<real_t>& f_next,
                     const std::vector<char>& obs, const std::vector<real_t>& halo_from_gpu,
                     std::vector<real_t>& halo_to_gpu_out, int ny_gpu) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = ny_gpu; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = y * NX + x;

            if (x == 0) {
                real_t u2 = u_inflow * u_inflow;
                for (int i = 0; i < 9; ++i) {
                    real_t cu = cx[i] * u_inflow;
                    f_next[i * NX * NY + idx] = w[i] * (REAL_LIT(1.0) + REAL_LIT(3.0) * cu + REAL_LIT(4.5) * cu * cu - REAL_LIT(1.5) * u2);
                }
                if (y == ny_gpu) for (int i = 0; i < 9; ++i) halo_to_gpu_out[i * NX + x] = f_next[i * NX * NY + idx];
                continue;
            }
            if (x == NX - 1) {
                int virtual_x = NX - 2;
                real_t g[9];
                for (int i = 0; i < 9; ++i) {
                    int prev_x = virtual_x - cx[i];
                    int prev_y = y - cy[i];
                    int v_idx = y * NX + virtual_x;
                    if (prev_y >= NY) {
                        g[i] = f[noslip[i] * NX * NY + v_idx];
                    } else if (prev_y < ny_gpu) {
                        g[i] = obs[prev_y * NX + prev_x] ? f[noslip[i] * NX * NY + v_idx]
                                                          : halo_from_gpu[i * NX + prev_x];
                    } else if (obs[prev_y * NX + prev_x]) {
                        g[i] = f[noslip[i] * NX * NY + v_idx];
                    } else {
                        g[i] = f[i * NX * NY + (prev_y * NX + prev_x)];
                    }
                }
                real_t rho = REAL_LIT(0.0), ux = REAL_LIT(0.0), uy = REAL_LIT(0.0);
                for (int i = 0; i < 9; ++i) { rho += g[i]; ux += g[i] * cx[i]; uy += g[i] * cy[i]; }
                ux /= rho; uy /= rho;
                real_t u2 = ux * ux + uy * uy;
                for (int i = 0; i < 9; ++i) {
                    real_t cu = cx[i] * ux + cy[i] * uy;
                    real_t feq = w[i] * rho * (REAL_LIT(1.0) + REAL_LIT(3.0) * cu + REAL_LIT(4.5) * cu * cu - REAL_LIT(1.5) * u2);
                    f_next[i * NX * NY + idx] = g[i] - (g[i] - feq) / tau;
                }
                if (y == ny_gpu) for (int i = 0; i < 9; ++i) halo_to_gpu_out[i * NX + x] = f_next[i * NX * NY + idx];
                continue;
            }
            if (obs[idx]) continue;

            real_t g[9];
            for (int i = 0; i < 9; ++i) g[i] = cpu_gather_dir(f, obs, halo_from_gpu, x, y, i, ny_gpu);
            real_t rho = REAL_LIT(0.0), ux = REAL_LIT(0.0), uy = REAL_LIT(0.0);
            for (int i = 0; i < 9; ++i) { rho += g[i]; ux += g[i] * cx[i]; uy += g[i] * cy[i]; }
            ux /= rho; uy /= rho;
            real_t u2 = ux * ux + uy * uy;
            for (int i = 0; i < 9; ++i) {
                real_t cu = cx[i] * ux + cy[i] * uy;
                real_t feq = w[i] * rho * (REAL_LIT(1.0) + REAL_LIT(3.0) * cu + REAL_LIT(4.5) * cu * cu - REAL_LIT(1.5) * u2);
                f_next[i * NX * NY + idx] = g[i] - (g[i] - feq) / tau;
            }
            if (y == ny_gpu) for (int i = 0; i < 9; ++i) halo_to_gpu_out[i * NX + x] = f_next[i * NX * NY + idx];
        }
    }
}

// ============================================================================
// NUEVO: mover filas de un lado a otro cuando cambia ny_gpu. El array guarda
// las 9 direcciones en 9 "planos" separados (f[i*NX*NY + fila*NX + x]), asi
// que copiar un rango de filas para las 9 direcciones a la vez es una copia
// con "huecos" (stride) -> cudaMemcpy2D, con height=9 (una fila de la
// "matriz 2D" por cada direccion) en vez de 9 cudaMemcpy sueltos.
// ============================================================================
void migrate_rows(real_t* d_f, std::vector<real_t>& h_f, int ny_gpu_old, int ny_gpu_new) {
    if (ny_gpu_new == ny_gpu_old) return;
    const size_t plane_stride = (size_t)NX * NY * sizeof(real_t);
    if (ny_gpu_new > ny_gpu_old) {
        // filas [ny_gpu_old, ny_gpu_new) pasan de CPU -> GPU
        int row0 = ny_gpu_old, nrows = ny_gpu_new - ny_gpu_old;
        cudaMemcpy2D(d_f + (size_t)row0 * NX, plane_stride,
                     h_f.data() + (size_t)row0 * NX, plane_stride,
                     (size_t)nrows * NX * sizeof(real_t), 9,
                     cudaMemcpyHostToDevice);
    } else {
        // filas [ny_gpu_new, ny_gpu_old) pasan de GPU -> CPU
        int row0 = ny_gpu_new, nrows = ny_gpu_old - ny_gpu_new;
        cudaMemcpy2D(h_f.data() + (size_t)row0 * NX, plane_stride,
                     d_f + (size_t)row0 * NX, plane_stride,
                     (size_t)nrows * NX * sizeof(real_t), 9,
                     cudaMemcpyDeviceToHost);
    }
}

// Tras mover la frontera, la fila de contacto ha cambiado -> hay que volver
// a rellenar los dos halos a partir del estado (ya consistente) que acaba
// de quedar en d_f/h_f, en vez de arrastrar el halo de la frontera vieja.
void refresh_halo_after_migration(const real_t* d_f, real_t* h_halo_gpu_pinned,
                                   const std::vector<real_t>& h_f, std::vector<real_t>& h_halo_cpu,
                                   int ny_gpu) {
    const size_t plane_stride = (size_t)NX * NY * sizeof(real_t);
    cudaMemcpy2D(h_halo_gpu_pinned, NX * sizeof(real_t),
                 d_f + (size_t)(ny_gpu - 1) * NX, plane_stride,
                 NX * sizeof(real_t), 9,
                 cudaMemcpyDeviceToHost);
    for (int i = 0; i < 9; ++i)
        std::memcpy(h_halo_cpu.data() + i * NX, h_f.data() + (size_t)i * NX * NY + (size_t)ny_gpu * NX,
                    NX * sizeof(real_t));
}

int main() {
    int ny_gpu = NY_GPU_INIT;

    std::vector<real_t> h_f(9 * NX * NY), h_f_next(9 * NX * NY);
    std::vector<char> h_obs(NX * NY, 0);

    int cx_cyl = NX / 4, cy_cyl = NY / 2, r_cyl = NY / 10;
    for (int y = 0; y < NY; ++y)
        for (int x = 0; x < NX; ++x)
            if ((x - cx_cyl)*(x - cx_cyl) + (y - cy_cyl)*(y - cy_cyl) < r_cyl * r_cyl)
                h_obs[y * NX + x] = 1;

    for (int y = 0; y < NY; ++y)
        for (int x = 0; x < NX; ++x) {
            int idx = y * NX + x;
            real_t u2_init = u_inflow * u_inflow;
            for (int i = 0; i < 9; ++i) {
                real_t cu = cx[i] * u_inflow;
                h_f[i * NX * NY + idx] = w[i] * (REAL_LIT(1.0) + REAL_LIT(3.0) * cu + REAL_LIT(4.5) * cu * cu - REAL_LIT(1.5) * u2_init);
            }
        }
    h_f_next = h_f;

    real_t *d_f, *d_f_next;
    bool *d_obstacle;
    real_t *d_halo_from_cpu, *d_halo_to_cpu_out;
    cudaMalloc(&d_f, 9 * NX * NY * sizeof(real_t));
    cudaMalloc(&d_f_next, 9 * NX * NY * sizeof(real_t));
    cudaMalloc(&d_obstacle, NX * NY * sizeof(bool));
    cudaMalloc(&d_halo_from_cpu, 9 * NX * sizeof(real_t));
    cudaMalloc(&d_halo_to_cpu_out, 9 * NX * sizeof(real_t));

    std::vector<char> h_obs_bool(NX * NY);
    for (int i = 0; i < NX * NY; ++i) h_obs_bool[i] = (h_obs[i] == 1);

    cudaMemcpy(d_f, h_f.data(), 9 * NX * NY * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obstacle, h_obs_bool.data(), NX * NY * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_cx, cx, 9 * sizeof(int));
    cudaMemcpyToSymbol(d_cy, cy, 9 * sizeof(int));
    cudaMemcpyToSymbol(d_w, w, 9 * sizeof(real_t));
    cudaMemcpyToSymbol(d_noslip, noslip, 9 * sizeof(int));

    real_t *h_halo_gpu_pinned;
    cudaMallocHost(&h_halo_gpu_pinned, 9 * NX * sizeof(real_t));
    std::vector<real_t> h_halo_cpu(9 * NX, REAL_LIT(0.0));

    dim3 blockSize(32, 8);
    dim3 gridSize((NX + blockSize.x - 1) / blockSize.x, (ny_gpu + blockSize.y - 1) / blockSize.y);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    std::cout << "=========================================================" << std::endl;
    std::cout << ">>> LBM D2Q9 HETEROGENEO - REBALANCEO DINAMICO" << std::endl;
    std::cout << "ny_gpu inicial = " << ny_gpu << " / NY = " << NY
              << "  (ventana de reequilibrio = " << REBALANCE_WINDOW << " pasos)" << std::endl;
    std::cout << "=========================================================" << std::endl;

    FILE* log_fp = fopen("rebalanceo_log.csv", "w");
    if (log_fp) fprintf(log_fp, "step,ny_gpu,t_gpu_us_por_fila,t_cpu_us_por_fila\n");

    auto start_time = std::chrono::high_resolution_clock::now();

    // --- Paso 0: colision sola en ambos lados (bootstrap) ---
    collision_only_het<<<gridSize, blockSize, 0, stream>>>(d_f, d_obstacle, d_halo_to_cpu_out, tau, ny_gpu);
    cpu_collision_only(h_f, h_obs, h_halo_cpu, ny_gpu);
    cudaMemcpyAsync(h_halo_gpu_pinned, d_halo_to_cpu_out, 9 * NX * sizeof(real_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // --- Pasos 1 .. NUM_STEPS-2: fusionados + rebalanceo periodico ---
    double gpu_ms_window = 0.0, cpu_ms_window = 0.0;
    int window_steps = 0;

    for (int step = 0; step < NUM_STEPS - 1; ++step) {
        std::vector<real_t> h_halo_gpu_prev_step(h_halo_gpu_pinned, h_halo_gpu_pinned + 9 * NX);
        std::vector<real_t> h_halo_cpu_prev_step(h_halo_cpu);

        cudaMemcpyAsync(d_halo_from_cpu, h_halo_cpu_prev_step.data(), 9 * NX * sizeof(real_t), cudaMemcpyHostToDevice, stream);
        cudaEventRecord(ev_start, stream);
        collide_stream_fused_het<<<gridSize, blockSize, 0, stream>>>(
            d_f, d_f_next, d_obstacle, d_halo_from_cpu, d_halo_to_cpu_out, tau, u_inflow, ny_gpu);
        cudaEventRecord(ev_stop, stream);
        cudaMemcpyAsync(h_halo_gpu_pinned, d_halo_to_cpu_out, 9 * NX * sizeof(real_t), cudaMemcpyDeviceToHost, stream);

        auto cpu_t0 = std::chrono::high_resolution_clock::now();
        cpu_fused_step(h_f, h_f_next, h_obs, h_halo_gpu_prev_step, h_halo_cpu, ny_gpu);
        auto cpu_t1 = std::chrono::high_resolution_clock::now();
        double cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

        cudaStreamSynchronize(stream);
        float gpu_ms = 0.0f;
        cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);

        std::swap(d_f, d_f_next);
        h_f.swap(h_f_next);

        gpu_ms_window += gpu_ms;
        cpu_ms_window += cpu_ms;
        window_steps++;

        if (window_steps == REBALANCE_WINDOW) {
            int ny_cpu = NY - ny_gpu;
            double t_gpu_row_ms = gpu_ms_window / window_steps / ny_gpu;
            double t_cpu_row_ms = cpu_ms_window / window_steps / ny_cpu;

            int ideal_ny_gpu = (int)std::lround(NY * t_cpu_row_ms / (t_gpu_row_ms + t_cpu_row_ms));
            ideal_ny_gpu = std::max(NY_GPU_MIN, std::min(NY_GPU_MAX, ideal_ny_gpu));

            int delta = ideal_ny_gpu - ny_gpu;
            delta = std::max(-MAX_STEP_ROWS, std::min(MAX_STEP_ROWS, delta));
            int new_ny_gpu = ny_gpu + delta;

            if (log_fp) {
                fprintf(log_fp, "%d,%d,%.4f,%.4f\n", step, ny_gpu, t_gpu_row_ms * 1000.0, t_cpu_row_ms * 1000.0);
            }

            if (new_ny_gpu != ny_gpu) {
                migrate_rows(d_f, h_f, ny_gpu, new_ny_gpu);
                ny_gpu = new_ny_gpu;
                gridSize.y = (ny_gpu + blockSize.y - 1) / blockSize.y;
                refresh_halo_after_migration(d_f, h_halo_gpu_pinned, h_f, h_halo_cpu, ny_gpu);
                std::cout << "  [reequilibrio] paso " << step << ": ny_gpu -> " << ny_gpu
                          << "  (GPU " << t_gpu_row_ms * 1000.0 << " us/fila, CPU "
                          << t_cpu_row_ms * 1000.0 << " us/fila)" << std::endl;
            }

            gpu_ms_window = 0.0; cpu_ms_window = 0.0; window_steps = 0;
        }
    }

    // --- Paso NUM_STEPS-1: streaming solo en ambos lados (unwrap final) ---
    cudaMemcpyAsync(d_halo_from_cpu, h_halo_cpu.data(), 9 * NX * sizeof(real_t), cudaMemcpyHostToDevice, stream);
    streaming_only_het<<<gridSize, blockSize, 0, stream>>>(d_f, d_f_next, d_obstacle, d_halo_from_cpu, u_inflow, ny_gpu);
    std::vector<real_t> h_halo_gpu_for_last(h_halo_gpu_pinned, h_halo_gpu_pinned + 9 * NX);
    cpu_streaming_only(h_f, h_f_next, h_obs, h_halo_gpu_for_last, ny_gpu);
    cudaStreamSynchronize(stream);
    std::swap(d_f, d_f_next);
    h_f.swap(h_f_next);

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(end_time - start_time).count();
    double mlups = (double(NX) * NY * NUM_STEPS) / (total_sec * 1e6);
    std::cout << "=========================================================" << std::endl;
    std::cout << "ny_gpu final = " << ny_gpu << " / NY = " << NY << std::endl;
    std::cout << "Tiempo Total: " << total_sec << " s. Rendimiento: " << mlups << " MLUPS" << std::endl;
    std::cout << "=========================================================" << std::endl;

    if (log_fp) fclose(log_fp);

    // --- Volcado para validar: recomponer el array COMPLETO (GPU + CPU) ---
    {
        std::vector<real_t> h_f_gpu_part(9 * NX * NY);
        cudaMemcpy(h_f_gpu_part.data(), d_f, 9 * NX * NY * sizeof(real_t), cudaMemcpyDeviceToHost);
        std::vector<real_t> h_f_final(9 * NX * NY);
        for (int i = 0; i < 9; ++i) {
            std::memcpy(h_f_final.data() + i * NX * NY, h_f_gpu_part.data() + i * NX * NY, ny_gpu * NX * sizeof(real_t));
            std::memcpy(h_f_final.data() + i * NX * NY + ny_gpu * NX, h_f.data() + i * NX * NY + ny_gpu * NX,
                        (NY - ny_gpu) * NX * sizeof(real_t));
        }
        FILE* fp = fopen("f_final.bin", "wb");
        if (fp) {
            fwrite(h_f_final.data(), sizeof(real_t), h_f_final.size(), fp);
            fclose(fp);
            std::cout << "Estado final volcado en f_final.bin (" << h_f_final.size() * sizeof(real_t) << " bytes)" << std::endl;
        }
    }

    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    cudaStreamDestroy(stream);
    cudaFree(d_f); cudaFree(d_f_next); cudaFree(d_obstacle);
    cudaFree(d_halo_from_cpu); cudaFree(d_halo_to_cpu_out);
    cudaFreeHost(h_halo_gpu_pinned);
    return 0;
}
