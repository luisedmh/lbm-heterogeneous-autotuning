// ============================================================================
// LBM D2Q9 - HETEROGENEO ESTATICO INSTRUMENTADO PARA BENCHMARK
//
// Es lbm_heterogeneo_v2.cu (reparto FIJO, halo bidireccional) SIN tocar la
// fisica ni el esquema numerico, con tres diferencias, todas de medida:
//
//   1) El reparto y la duracion se eligen por linea de comandos, sin
//      recompilar:   --ny-gpu N   --steps N   --warmup N   --dump
//   2) En cada paso se mide POR SEPARADO cuanto tarda el lado GPU
//      (cudaEvent: copia del halo H2D + kernel + copia del halo D2H) y el
//      lado CPU (cpu_fused_step, con reloj del host), ademas del paso entero.
//      Los primeros --warmup pasos se descartan de las estadisticas.
//   3) Al final imprime UNA linea maquina-legible que empieza por
//      BENCH_RESULT, que es la que lee scripts/benchmark.py. El volcado de
//      f_final.bin (72 MB en FP64) solo se hace con --dump.
//
// Para validar que la instrumentacion no ha roto nada: ejecutar con --dump
// (3000 pasos, FP64) y comparar con compara_resultados.py como siempre.
//
// ---- Cabecera original de lbm_heterogeneo_v2.cu ----
// LBM D2Q9 - HETEROGENEO GPU+CPU v2 - reparto FIJO todavia (paso previo al
// reparto dinamico). Corrige dos bugs de tu version original:
//
//   BUG 1 (carrera en CPU): tu cpu_lbm_step colisionaba y hacia streaming de
//   la misma celda en el mismo bucle paralelo, leyendo datos de celdas
//   vecinas que otro hilo podia no haber colisionado todavia. Aqui, igual
//   que en la GPU, la CPU usa el patron "recoger (de un buffer CONGELADO,
//   el del paso anterior) y luego relajar" -> nunca lee nada que otro hilo
//   este escribiendo a la vez, sin condicion de carrera posible.
//
//   BUG 2 (halo de un solo sentido): tu kernel de GPU trataba la fila de
//   contacto con la CPU como una pared para las direcciones que "vienen"
//   desde la CPU. Aqui el halo viaja en los dos sentidos: cada lado manda
//   al otro su fila de contacto (resultado de ESTE paso) para que el otro
//   la use como entrada en el paso SIGUIENTE. Eso es tambien lo que permite
//   que GPU y CPU trabajen de verdad en paralelo: cada uno solo necesita el
//   halo del paso anterior del otro, nunca el de este mismo paso.
//
// ESTRUCTURA (identica en ambos lados, igual que lbm_gpu_fused.cu):
//   1) colision sola una vez al principio (bootstrap)
//   2) recoger+relajar fusionados, NUM_STEPS-1 veces (los pasos intermedios)
//   3) streaming solo una vez al final (desenvolver al estado fisico final)
//
// MEMORIA: tanto GPU como CPU reservan el dominio COMPLETO (los NY
// renglones); "ny_gpu" es solo un indice que marca la frontera, para poder
// cambiarlo sin reservar memoria de nuevo cuando montemos el reparto
// dinamico.
//
// VALIDACION: vuelca el estado final a f_final_heterogeneo.bin con el mismo
// layout que lbm_gpu_fused.cu, para comparar con compara_resultados.py
// contra una ejecucion de referencia de lbm_gpu_fused.cu sobre el dominio
// COMPLETO sin repartir nada.
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstring>
#include <omp.h>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <cuda_runtime.h>

const int NX = 1000;
const int NY = 1000;
const int NUM_STEPS_DEFAULT = 3000;

// Reparto FIJO de momento (90% GPU / 10% CPU). El valor exacto no afecta a
// la validacion de correccion (hasta un 50/50 validaria igual de bien).
const int NY_GPU_INIT = 900;

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
// (y=0) y de la frontera con la CPU (ny_gpu). Para celdas interiores.
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
        // solo ocurre en y == ny_gpu-1 (fila de contacto con la CPU)
        if (obstacle_full[prev_y * NX + prev_x]) return f[d_noslip[i] * N + idx];
        else return halo_from_cpu[i * NX + prev_x];
    } else if (obstacle_full[prev_y * NX + prev_x]) {
        return f[d_noslip[i] * N + idx];
    } else {
        return f[i * N + (prev_y * NX + prev_x)];
    }
}

// KERNEL A (GPU): colision sola. Se usa UNA vez, al principio (bootstrap).
// No hace gather (no necesita halo de entrada), pero SI empaqueta su fila
// de contacto para que la CPU la use como halo en su primer paso fusionado.
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

// KERNEL B (GPU): streaming solo. Se usa UNA vez, al final (unwrap).
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

// KERNEL C (GPU): recoger+relajar fusionados. Para todos los pasos
// intermedios. Necesita halo de entrada Y produce halo de salida.
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
// CPU: gather de una direccion, consciente de la pared inferior (y=NY-1) y
// de la frontera con la GPU (ny_gpu).
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
        // solo ocurre en y == ny_gpu (primera fila de la CPU)
        if (obs[prev_y * NX + prev_x]) return f[noslip[i] * NX * NY + idx];
        else return halo_from_gpu[i * NX + prev_x];
    } else if (obs[prev_y * NX + prev_x]) {
        return f[noslip[i] * NX * NY + idx];
    } else {
        return f[i * NX * NY + (prev_y * NX + prev_x)];
    }
}

// FASE A (CPU): colision sola, in-place. Se usa UNA vez, al principio
// (bootstrap) - simetrico a collision_only_het.
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

// FASE B (CPU): streaming solo. Se usa UNA vez, al final (unwrap) -
// simetrico a streaming_only_het.
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

// FASE C (CPU): recoger+relajar fusionados. Para todos los pasos
// intermedios - simetrico a collide_stream_fused_het. Lee SOLO del buffer
// congelado (f, del paso anterior) y escribe SOLO en f_next -> ningun hilo
// lee nunca algo que otro hilo este escribiendo, sin condicion de carrera.
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

// Estadisticos de un vector de tiempos (ms)
struct Stats { double mean, std, median, mn, mx; };
static Stats compute_stats(std::vector<double> v) {
    Stats s{0, 0, 0, 0, 0};
    if (v.empty()) return s;
    double sum = 0.0;
    for (double x : v) sum += x;
    s.mean = sum / v.size();
    double acc = 0.0;
    for (double x : v) acc += (x - s.mean) * (x - s.mean);
    s.std = std::sqrt(acc / v.size());
    std::sort(v.begin(), v.end());
    s.median = v[v.size() / 2];
    s.mn = v.front();
    s.mx = v.back();
    return s;
}

int main(int argc, char** argv) {
    int ny_gpu = NY_GPU_INIT;
    int NUM_STEPS = NUM_STEPS_DEFAULT;
    int warmup_steps = 100;
    bool do_dump = false;
    for (int a = 1; a < argc; ++a) {
        std::string arg = argv[a];
        if (arg == "--ny-gpu" && a + 1 < argc)      ny_gpu = std::atoi(argv[++a]);
        else if (arg == "--steps" && a + 1 < argc)  NUM_STEPS = std::atoi(argv[++a]);
        else if (arg == "--warmup" && a + 1 < argc) warmup_steps = std::atoi(argv[++a]);
        else if (arg == "--dump")                   do_dump = true;
        else { std::cerr << "Argumento desconocido: " << arg << std::endl; return 2; }
    }
    if (ny_gpu < 2 || ny_gpu > NY - 2) {
        std::cerr << "ny_gpu fuera de rango (2.." << NY - 2 << "): " << ny_gpu << std::endl;
        return 2;
    }
    if (NUM_STEPS < 3) { std::cerr << "--steps debe ser >= 3" << std::endl; return 2; }
    if (warmup_steps >= NUM_STEPS - 1) warmup_steps = (NUM_STEPS - 1) / 4;

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

    real_t *h_halo_gpu_pinned; // halo GPU->CPU, en memoria fijada (pinned)
    cudaMallocHost(&h_halo_gpu_pinned, 9 * NX * sizeof(real_t));
    std::vector<real_t> h_halo_cpu(9 * NX, REAL_LIT(0.0)); // halo CPU->GPU (en host, normal)

    dim3 blockSize(32, 8);
    dim3 gridSize((NX + blockSize.x - 1) / blockSize.x, (ny_gpu + blockSize.y - 1) / blockSize.y);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::cout << "=========================================================" << std::endl;
    std::cout << ">>> LBM D2Q9 HETEROGENEO ESTATICO - BENCHMARK (reparto fijo, halo bidireccional)" << std::endl;
    std::cout << "ny_gpu = " << ny_gpu << " / NY = " << NY << std::endl;
    std::cout << "=========================================================" << std::endl;

    // --- instrumentacion: eventos CUDA y vectores de tiempos por paso (ms) ---
    cudaEvent_t ev_g0, ev_g1;
    cudaEventCreate(&ev_g0);
    cudaEventCreate(&ev_g1);
    std::vector<double> v_step, v_gpu, v_cpu;
    v_step.reserve(NUM_STEPS); v_gpu.reserve(NUM_STEPS); v_cpu.reserve(NUM_STEPS);
    using bclk = std::chrono::steady_clock;

    auto start_time = std::chrono::high_resolution_clock::now();

    // --- Paso 0: colision sola en ambos lados (bootstrap), sin halo de entrada ---
    collision_only_het<<<gridSize, blockSize, 0, stream>>>(d_f, d_obstacle, d_halo_to_cpu_out, tau, ny_gpu);
    cpu_collision_only(h_f, h_obs, h_halo_cpu, ny_gpu);
    cudaMemcpyAsync(h_halo_gpu_pinned, d_halo_to_cpu_out, 9 * NX * sizeof(real_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream); // aqui SI hace falta esperar: ambos halos deben estar listos

    // --- Pasos 1 .. NUM_STEPS-2: fusionados, halo del paso anterior en ambos lados ---
    for (int step = 0; step < NUM_STEPS - 1; ++step) {
        auto t_step0 = bclk::now();
        // 1) Capturamos AHORA, de forma sincrona en el host, una copia de los
        //    halos que dejaron listos GPU y CPU en el paso anterior -> ANTES
        //    de pedirle a nadie que los sobreescriba. Esto es imprescindible:
        //    si leyeramos h_halo_gpu_pinned/h_halo_cpu directamente mas abajo
        //    (despues de lanzar las copias/el calculo que los sobreescribe),
        //    habria una carrera de verdad entre esa lectura y la escritura
        //    asincrona de la GPU (el cudaMemcpyAsync D2H se ejecuta en cuanto
        //    el kernel termina, NO cuando llega el cudaStreamSynchronize; eso
        //    solo bloquea al HOST, no decide cuando trabaja la GPU). Haciendo
        //    la copia aqui, antes de tocar nada, nos aseguramos de que nadie
        //    escribe estos dos buffers mientras se leen.
        std::vector<real_t> h_halo_gpu_prev_step(h_halo_gpu_pinned, h_halo_gpu_pinned + 9 * NX);
        std::vector<real_t> h_halo_cpu_prev_step(h_halo_cpu); // h_halo_cpu se sobreescribe mas abajo

        // 2) subir a la GPU el halo de la CPU del paso anterior (copia congelada, segura)
        cudaEventRecord(ev_g0, stream);   // [BENCH] inicio del lado GPU
        cudaMemcpyAsync(d_halo_from_cpu, h_halo_cpu_prev_step.data(), 9 * NX * sizeof(real_t), cudaMemcpyHostToDevice, stream);
        // 3) lanzar el kernel fusionado en la GPU (async: el host no espera aqui)
        collide_stream_fused_het<<<gridSize, blockSize, 0, stream>>>(
            d_f, d_f_next, d_obstacle, d_halo_from_cpu, d_halo_to_cpu_out, tau, u_inflow, ny_gpu);
        // 4) pedir la copia async GPU->host del halo NUEVO que acaba de calcular el kernel.
        //    Esto sobreescribira h_halo_gpu_pinned en cuanto el kernel del punto 3 termine,
        //    pero eso pasa DESPUES de que el punto 1 ya termino de leerlo (mismo hilo, en
        //    orden), asi que nunca se lee y se escribe el mismo buffer a la vez.
        cudaMemcpyAsync(h_halo_gpu_pinned, d_halo_to_cpu_out, 9 * NX * sizeof(real_t), cudaMemcpyDeviceToHost, stream);

        cudaEventRecord(ev_g1, stream);   // [BENCH] fin del lado GPU
        auto t_cpu0 = bclk::now();
        // 5) mientras la GPU trabaja (puntos 3-4, en su propio stream), la CPU calcula su
        //    propio paso YA en paralelo de verdad, usando las copias congeladas del punto 1
        //    (los halos del paso ANTERIOR, como exige el esquema recoger+relajar)
        cpu_fused_step(h_f, h_f_next, h_obs, h_halo_gpu_prev_step, h_halo_cpu, ny_gpu);
        auto t_cpu1 = bclk::now();

        cudaStreamSynchronize(stream);
        std::swap(d_f, d_f_next);
        h_f.swap(h_f_next);
        auto t_step1 = bclk::now();

        if (step >= warmup_steps) {   // [BENCH] los primeros pasos no cuentan
            float ms_gpu = 0.0f;
            cudaEventElapsedTime(&ms_gpu, ev_g0, ev_g1);
            v_gpu.push_back((double)ms_gpu);
            v_cpu.push_back(std::chrono::duration<double, std::milli>(t_cpu1 - t_cpu0).count());
            v_step.push_back(std::chrono::duration<double, std::milli>(t_step1 - t_step0).count());
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
    std::cout << "Tiempo Total: " << total_sec << " s. Rendimiento: " << mlups << " MLUPS" << std::endl;

    // --- [BENCH] resumen maquina-legible ---
    {
        Stats ss = compute_stats(v_step), sg = compute_stats(v_gpu), sc = compute_stats(v_cpu);
        int ny_cpu = NY - ny_gpu;
        // MLUPS "del lado": celdas de ese lado / tiempo medio que ese lado tarda en un paso
        double gpu_side_mlups = (double)NX * ny_gpu / (sg.mean * 1e3);
        double cpu_side_mlups = (double)NX * ny_cpu / (sc.mean * 1e3);
        const char* prec =
#ifdef USE_SINGLE_PRECISION
            "FP32";
#else
            "FP64";
#endif
        std::cout << "BENCH_RESULT"
                  << " precision=" << prec
                  << " NX=" << NX << " NY=" << NY
                  << " ny_gpu=" << ny_gpu << " ny_cpu=" << ny_cpu
                  << " steps=" << NUM_STEPS << " warmup=" << warmup_steps
                  << " n_samples=" << v_step.size()
                  << " omp_threads=" << omp_get_max_threads()
                  << " t_total_s=" << total_sec
                  << " mlups_total=" << mlups
                  << " t_step_ms_mean=" << ss.mean << " t_step_ms_std=" << ss.std << " t_step_ms_median=" << ss.median
                  << " t_gpu_ms_mean=" << sg.mean << " t_gpu_ms_std=" << sg.std << " t_gpu_ms_median=" << sg.median
                  << " t_cpu_ms_mean=" << sc.mean << " t_cpu_ms_std=" << sc.std << " t_cpu_ms_median=" << sc.median
                  << " gpu_side_mlups=" << gpu_side_mlups
                  << " cpu_side_mlups=" << cpu_side_mlups
                  << std::endl;
    }

    // --- Volcado para validar: recomponer el array COMPLETO (GPU + CPU) ---
    if (do_dump) {
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

    cudaEventDestroy(ev_g0); cudaEventDestroy(ev_g1);
    cudaStreamDestroy(stream);
    cudaFree(d_f); cudaFree(d_f_next); cudaFree(d_obstacle);
    cudaFree(d_halo_from_cpu); cudaFree(d_halo_to_cpu_out);
    cudaFreeHost(h_halo_gpu_pinned);
    return 0;
}
