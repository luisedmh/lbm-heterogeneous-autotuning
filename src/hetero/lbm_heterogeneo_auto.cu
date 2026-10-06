// ============================================================================
// LBM D2Q9 - HETEROGENEO "AUTO": decide SOLO, con numeros medidos en la
// propia maquina, si merece la pena usar la CPU o si el overhead de
// sincronizacion se come cualquier ganancia posible - y en ese caso colapsa
// a GPU pura (sin pagar ni un microsegundo de mas por CPU/halos que no van
// a servir de nada). Funciona igual en fp64 que en fp32: no hay ningun
// numero de precision cableado, todo sale de medir.
//
// LA IDEA, EN UNA FRASE: en vez de asumir que "fp64 siempre compensa y fp32
// nunca", el programa mide sus propios tiempos por fila (GPU y CPU) y su
// propio overhead fijo, calcula el tamano de problema minimo a partir del
// cual la CPU empezaria a ayudar, y compara eso con el tamano real del
// problema que le has dado. Los tres factores que ya vimos que mueven esa
// frontera -precision, tamano del problema, coste del operador de colision-
// quedan todos capturados en dos numeros medidos (cuanto tarda la GPU por
// fila, cuanto tarda la CPU por fila) mas uno (el overhead fijo), sin que
// haga falta saber de antemano en que caso estas.
//
// LA MATEMATICA (se explica tambien en el mensaje que acompaña a este
// archivo, aqui el resumen para quien lea el codigo):
//
//   t_gpu   = tiempo de GPU por fila (colision+streaming fusionados)
//   t_cpu   = tiempo de CPU por fila (idem, ya incluye el propio overhead
//             de arrancar los hilos OpenMP en cada paso)
//   T_ovh   = overhead FIJO por paso que solo paga el esquema heterogeneo
//             (lanzar el kernel, copiar el halo en los dos sentidos,
//             sincronizar el stream) - no depende de cuantas filas reparte
//             cada lado, asi que en microsegundos es (casi) constante.
//
//   Reparto que iguala los dos tiempos (sin contar el overhead):
//       ny_gpu* = NY * t_cpu / (t_gpu + t_cpu)
//
//   Margen que se gana repartiendo en vez de ir toda a GPU (sin overhead):
//       margen(NY) = NY * t_gpu^2 / (t_gpu + t_cpu)
//
//   Compensa usar la CPU si y solo si margen(NY) > T_ovh, lo que equivale a
//       NY > NY_critico = T_ovh * (t_gpu + t_cpu) / t_gpu^2
//
//   (con un margen de seguridad SAFETY_FACTOR para no cambiar de estrategia
//   por una diferencia dentro del ruido de medir).
//
// COMO SE MIDEN t_gpu, t_cpu y T_ovh: una ventana corta (CAL_WINDOW pasos)
// al principio, con un reparto de calibracion razonable (90% GPU / 10% CPU),
// usando EXACTAMENTE los mismos kernels/funciones heterogeneos que ya
// validamos en lbm_heterogeneo_dinamico.cu. t_gpu sale del cudaEvent
// (tiempo puro del kernel), t_cpu del std::chrono alrededor de
// cpu_fused_step (tiempo puro de esa llamada, overhead de sus propios
// hilos OpenMP incluido), y T_ovh de restarle al tiempo de pared del paso
// completo el mayor de los dos anteriores.
//
// QUE PASA DESPUES DE DECIDIR:
//   - Si compensa: se migran las filas al reparto ideal ny_gpu* (con
//     cudaMemcpy2D, como en la version dinamica) y se sigue exactamente con
//     el mismo bucle de reequilibrio dinamico de lbm_heterogeneo_dinamico.cu
//     para el resto de la simulacion.
//   - Si NO compensa: se migran TODAS las filas de vuelta a la GPU y, a
//     partir de ahi, se usan los kernels de SOLO GPU de lbm_gpu_fused.cu
//     (ya validados, sin ningun concepto de ny_gpu ni halo) para el resto
//     de pasos - cero llamadas a CPU, cero copias de halo, cero overhead
//     heterogeneo a partir de ese momento.
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

// Comprobacion de errores CUDA. Sin esto, un fallo real (tipico: sin
// memoria de GPU suficiente para NX*NY grande) NO para el programa ni
// avisa de nada - cudaMalloc devuelve un puntero invalido, los kernels
// lanzados sobre el se saltan silenciosamente (no hacen ningun trabajo
// real), y el resto del codigo sigue como si nada, produciendo numeros
// de rendimiento absurdamente altos (porque "computar sobre nada" es
// instantaneo) y un fichero de salida con datos sin sentido, sin ningun
// mensaje de error visible. Con este macro, cualquier fallo de este tipo
// para el programa inmediatamente con un mensaje claro.
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        std::cerr << "ERROR CUDA en " << __FILE__ << ":" << __LINE__ << " -> " \
                  << cudaGetErrorString(_e) << std::endl; \
        std::exit(1); \
    } \
} while (0)

const int NX = 1000;
const int NY = 1000;
const int NUM_STEPS = 3000;

// Ventana de calibracion (pasos) y de reequilibrio dinamico posterior.
const int CAL_WINDOW = 80;
const int REBALANCE_WINDOW = 30;
const int MAX_STEP_ROWS = 120;

// Pasos de "calentamiento" ANTES de empezar a medir de verdad la
// calibracion, sin contar para la media. Sin esto, la GPU todavia esta
// subiendo de reloj (boost/DVFS) en los primeros lanzamientos tras arrancar
// el proceso, y t_gpu sale mas lento de lo que rinde de verdad en regimen
// permanente -> la formula subestima a la GPU, le da mas trabajo del que
// deberia a la CPU, y el reparto sale peor de lo que promete el calculo.
const int WARMUP_STEPS = 30;

// Banda muerta del reequilibrio dinamico: si el reparto "ideal" recalculado
// difiere del actual en menos de esto, NO se mueve nada. Migrar filas tiene
// un coste (cudaMemcpy2D + rehacer el halo) que no compensa corregir un
// desequilibrio de un puñado de filas -> sin esto, el programa se pasa la
// simulacion migrando por ruido de medida sin ganar nada real a cambio.
const int MIN_STEP_ROWS = 25;

// Limites de seguridad para no acercarse a los casos ny_gpu=0 / ny_gpu=NY
// (documentados en lbm_heterogeneo_v2.cu / lbm_heterogeneo_dinamico.cu).
const int NY_GPU_MIN = 20;
const int NY_GPU_MAX = NY - 20;

// Si el dominio ni siquiera deja hueco para una calibracion razonable en
// los dos lados, ni lo intentamos: vamos directos a GPU sola.
const int MIN_NY_FOR_HET_CONSIDERATION = 3 * NY_GPU_MIN;

// Pasos de calentamiento + medicion para calibrar la GPU PURA (el kernel de
// lbm_gpu_fused.cu, sin ningun concepto de ny_gpu/halo). Hace falta medirla
// por separado: el kernel heterogeneo, aunque haga la misma fisica, tiene
// una rama extra (comprobar si una direccion viene del halo de la CPU) y
// parametros de mas que el kernel puro no necesita, y eso le cuesta un
// tiempo real por celda - no es un sesgo de medida, es un coste estructural
// de "saber hablar con la CPU" que paga aunque la CPU no este haciendo nada.
const int PURE_WARMUP_STEPS = 15;
const int PURE_CAL_STEPS = 50;

// Exigimos que el margen total ganado repartiendo (NY * margen_por_fila)
// supere el overhead fijo del paso heterogeneo multiplicado por este factor
// -> evita cambiar de estrategia por una diferencia que puede ser solo
// ruido de medida (1.1 = exigir un margen al menos un ~10% mayor que el
// overhead fijo medido). Ver el bloque de decision mas abajo (usar_heterogeneo).
const double SAFETY_FACTOR = 1.1;

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
// KERNELS DE SOLO-GPU (identicos a lbm_gpu_fused.cu, sin ny_gpu ni halo:
// operan sobre el dominio COMPLETO). Se usan cuando la calibracion decide
// que no compensa repartir con la CPU.
// ============================================================================
__global__ void collision_only_kernel(real_t* __restrict__ f, const bool* __restrict__ obstacle, real_t tau_param) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;
    int idx = y * NX + x;
    const int N = NX * NY;
    if (obstacle[idx]) return;

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
}

__global__ void streaming_only_kernel(const real_t* __restrict__ f, real_t* __restrict__ f_next,
                                       const bool* __restrict__ obstacle, real_t u_inflow_param) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;
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
            if (prev_y < 0 || prev_y >= NY || obstacle[prev_y * NX + prev_x]) {
                f_next[i * N + idx] = f[d_noslip[i] * N + v_idx];
            } else {
                f_next[i * N + idx] = f[i * N + (prev_y * NX + prev_x)];
            }
        }
        return;
    }
    if (obstacle[idx]) return;

    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        int prev_x = x - d_cx[i];
        int prev_y = y - d_cy[i];
        int prev_idx = prev_y * NX + prev_x;
        if (prev_y < 0 || prev_y >= NY || obstacle[prev_idx]) {
            f_next[i * N + idx] = f[d_noslip[i] * N + idx];
        } else {
            f_next[i * N + idx] = f[i * N + prev_idx];
        }
    }
}

__global__ void collide_stream_fused_kernel(const real_t* __restrict__ f, real_t* __restrict__ f_next,
                                             const bool* __restrict__ obstacle,
                                             real_t tau_param, real_t u_inflow_param) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;
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
        real_t g[9];
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            int prev_x = virtual_x - d_cx[i];
            int prev_y = y - d_cy[i];
            if (prev_y < 0 || prev_y >= NY || obstacle[prev_y * NX + prev_x]) {
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
        return;
    }
    if (obstacle[idx]) return;

    real_t g[9];
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        int prev_x = x - d_cx[i];
        int prev_y = y - d_cy[i];
        if (prev_y < 0 || prev_y >= NY || obstacle[prev_y * NX + prev_x]) {
            g[i] = f[d_noslip[i] * N + idx];
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
}

// ============================================================================
// KERNELS/FUNCIONES HETEROGENEOS (identicos a lbm_heterogeneo_dinamico.cu).
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

void migrate_rows(real_t* d_f, std::vector<real_t>& h_f, int ny_gpu_old, int ny_gpu_new) {
    if (ny_gpu_new == ny_gpu_old) return;
    const size_t plane_stride = (size_t)NX * NY * sizeof(real_t);
    if (ny_gpu_new > ny_gpu_old) {
        int row0 = ny_gpu_old, nrows = ny_gpu_new - ny_gpu_old;
        cudaMemcpy2D(d_f + (size_t)row0 * NX, plane_stride,
                     h_f.data() + (size_t)row0 * NX, plane_stride,
                     (size_t)nrows * NX * sizeof(real_t), 9,
                     cudaMemcpyHostToDevice);
    } else {
        int row0 = ny_gpu_new, nrows = ny_gpu_old - ny_gpu_new;
        cudaMemcpy2D(h_f.data() + (size_t)row0 * NX, plane_stride,
                     d_f + (size_t)row0 * NX, plane_stride,
                     (size_t)nrows * NX * sizeof(real_t), 9,
                     cudaMemcpyDeviceToHost);
    }
}

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
    {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        double needed_gb = 2.0 * 9.0 * (double)NX * (double)NY * sizeof(real_t) / 1e9;
        std::cout << "Memoria GPU: " << (total_b / 1e9) << " GB totales, "
                  << (free_b / 1e9) << " GB libres. d_f+d_f_next necesitan ~"
                  << needed_gb << " GB." << std::endl;
    }
    CUDA_CHECK(cudaMalloc(&d_f, 9 * NX * NY * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&d_f_next, 9 * NX * NY * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&d_obstacle, NX * NY * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_halo_from_cpu, 9 * NX * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc(&d_halo_to_cpu_out, 9 * NX * sizeof(real_t)));

    std::vector<char> h_obs_bool(NX * NY);
    for (int i = 0; i < NX * NY; ++i) h_obs_bool[i] = (h_obs[i] == 1);

    cudaMemcpy(d_f, h_f.data(), 9 * NX * NY * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obstacle, h_obs_bool.data(), NX * NY * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_cx, cx, 9 * sizeof(int));
    cudaMemcpyToSymbol(d_cy, cy, 9 * sizeof(int));
    cudaMemcpyToSymbol(d_w, w, 9 * sizeof(real_t));
    cudaMemcpyToSymbol(d_noslip, noslip, 9 * sizeof(int));

    real_t *h_halo_gpu_pinned;
    CUDA_CHECK(cudaMallocHost(&h_halo_gpu_pinned, 9 * NX * sizeof(real_t)));
    std::vector<real_t> h_halo_cpu(9 * NX, REAL_LIT(0.0));

    dim3 blockSize(32, 8);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

#ifdef USE_SINGLE_PRECISION
    std::cout << "Precision: FP32 (float)" << std::endl;
#else
    std::cout << "Precision: FP64 (double)" << std::endl;
#endif
    std::cout << "=========================================================" << std::endl;
    std::cout << ">>> LBM D2Q9 HETEROGENEO AUTO (decide GPU-sola vs GPU+CPU midiendo)" << std::endl;
    std::cout << "=========================================================" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    bool intentar_heterogeneo = (NY >= MIN_NY_FOR_HET_CONSIDERATION);
    bool usar_heterogeneo = false;   // decision final, se rellena mas abajo
    int ny_gpu = NY;                 // arranca asumiendo GPU sola
    int pasos_restantes = NUM_STEPS - 1; // pasos "fusionados" que quedan por hacer

    // Bootstrap UNICO de toda la simulacion (colision sola, en el dominio
    // completo, con el kernel de GPU pura). A partir de aqui el buffer
    // siempre representa "resultado de una colision" y ese invariante vale
    // igual para el kernel puro que para el heterogeneo - no hace falta
    // repetir el bootstrap cada vez que cambiamos de kernel.
    dim3 gridSizeFull((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y);
    collision_only_kernel<<<gridSizeFull, blockSize>>>(d_f, d_obstacle, tau);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    if (!intentar_heterogeneo) {
        std::cout << "NY=" << NY << " es demasiado pequeno para plantearse repartir (minimo "
                  << MIN_NY_FOR_HET_CONSIDERATION << "). Voy directo a GPU sola." << std::endl;
    } else {
        // --- Paso 1: calibrar la GPU PURA (linea base real a batir) ---
        double gpu_pure_ms_sum = 0.0;
        for (int step = 0; step < PURE_WARMUP_STEPS + PURE_CAL_STEPS; ++step) {
            cudaEventRecord(ev_start);
            collide_stream_fused_kernel<<<gridSizeFull, blockSize>>>(d_f, d_f_next, d_obstacle, tau, u_inflow);
            cudaEventRecord(ev_stop);
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            CUDA_CHECK(cudaGetLastError());
            std::swap(d_f, d_f_next);
            if (step >= PURE_WARMUP_STEPS) {
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev_start, ev_stop);
                gpu_pure_ms_sum += ms;
            }
        }
        pasos_restantes -= (PURE_WARMUP_STEPS + PURE_CAL_STEPS);
        double t_gpu_pure = gpu_pure_ms_sum / PURE_CAL_STEPS / NY;   // ms/fila, kernel puro, dominio completo

        // --- Paso 2: repartir el dominio actual (90/10) y calibrar el esquema HETEROGENEO ---
        int ny_cpu_cal = std::max(NY_GPU_MIN, (int)std::lround(0.10 * NY));
        int ny_gpu_cal = NY - ny_cpu_cal;
        ny_gpu_cal = std::max(NY_GPU_MIN, std::min(NY_GPU_MAX, ny_gpu_cal));

        // El buffer ya es valido (post-colision) desde el bootstrap de mas
        // arriba, y ahora mismo esta TODO en d_f (venimos de la calibracion
        // de GPU pura) -> migrar la parte que le toca a la CPU y rehacer los
        // halos para que reflejen esta nueva frontera antes de medir nada.
        migrate_rows(d_f, h_f, NY, ny_gpu_cal);
        ny_gpu = ny_gpu_cal;
        refresh_halo_after_migration(d_f, h_halo_gpu_pinned, h_f, h_halo_cpu, ny_gpu);

        dim3 gridSizeCal((NX + blockSize.x - 1) / blockSize.x, (ny_gpu + blockSize.y - 1) / blockSize.y);

        double gpu_ms_sum = 0.0, cpu_ms_sum = 0.0, wall_ms_sum = 0.0;

        // WARMUP_STEPS primero (sin medir, solo para que la GPU llegue a su
        // reloj de regimen permanente) + CAL_WINDOW despues (esos si cuentan
        // para la media). Un unico bucle, se distingue por el indice.
        for (int step = 0; step < WARMUP_STEPS + CAL_WINDOW; ++step) {
            bool medir = (step >= WARMUP_STEPS);
            auto wall_t0 = std::chrono::high_resolution_clock::now();

            std::vector<real_t> h_halo_gpu_prev_step(h_halo_gpu_pinned, h_halo_gpu_pinned + 9 * NX);
            std::vector<real_t> h_halo_cpu_prev_step(h_halo_cpu);

            cudaMemcpyAsync(d_halo_from_cpu, h_halo_cpu_prev_step.data(), 9 * NX * sizeof(real_t), cudaMemcpyHostToDevice, stream);
            cudaEventRecord(ev_start, stream);
            collide_stream_fused_het<<<gridSizeCal, blockSize, 0, stream>>>(
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

            auto wall_t1 = std::chrono::high_resolution_clock::now();
            double wall_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();

            if (medir) { gpu_ms_sum += gpu_ms; cpu_ms_sum += cpu_ms; wall_ms_sum += wall_ms; }
        }
        pasos_restantes -= (WARMUP_STEPS + CAL_WINDOW);

        double t_gpu_het = gpu_ms_sum / CAL_WINDOW / ny_gpu_cal;          // ms/fila, kernel HETEROGENEO
        double t_cpu = cpu_ms_sum / CAL_WINDOW / (NY - ny_gpu_cal);       // ms/fila
        double t_wall = wall_ms_sum / CAL_WINDOW;                        // ms/paso, tiempo real de pared
        double t_ovh = std::max(0.0, t_wall - std::max(ny_gpu_cal * t_gpu_het, (NY - ny_gpu_cal) * t_cpu));

        // Tiempo de referencia real a batir: GPU PURA sobre el dominio
        // completo (no el heterogeneo con ny_gpu=NY, que seria mas lento
        // por la rama/parametros de mas que no necesita si no hay CPU).
        double T_baseline = (double)NY * t_gpu_pure;
        // Tiempo previsto del esquema heterogeneo en su reparto optimo
        // interno (el que iguala GPU-heterogenea y CPU), mas el overhead fijo.
        double T_predicted_het = (double)NY * t_gpu_het * t_cpu / (t_gpu_het + t_cpu) + t_ovh;

        std::cout << "--- Calibracion GPU pura (" << PURE_CAL_STEPS << " pasos) ---" << std::endl;
        std::cout << "  t_gpu_pura = " << t_gpu_pure * 1000.0 << " us/fila" << std::endl;
        std::cout << "--- Calibracion heterogenea (" << CAL_WINDOW << " pasos, reparto " << ny_gpu_cal << "/" << (NY - ny_gpu_cal) << ") ---" << std::endl;
        std::cout << "  t_gpu_heterogeneo = " << t_gpu_het * 1000.0 << " us/fila  (coste extra por saber hablar con la CPU: "
                  << (t_gpu_het / t_gpu_pure - 1.0) * 100.0 << "%)" << std::endl;
        std::cout << "  t_cpu = " << t_cpu * 1000.0 << " us/fila" << std::endl;
        std::cout << "  overhead fijo medido ~ " << t_ovh * 1000.0 << " us/paso" << std::endl;
        std::cout << "  tiempo GPU pura (referencia, NY=" << NY << ") : " << T_baseline * 1000.0 << " us/paso" << std::endl;
        std::cout << "  tiempo heterogeneo previsto (reparto optimo)   : " << T_predicted_het * 1000.0 << " us/paso" << std::endl;

        double denom = t_gpu_pure - (t_gpu_het * t_cpu / (t_gpu_het + t_cpu));
        if (denom > 0.0) {
            double ny_critico = t_ovh / denom;
            std::cout << "  NY critico (a partir de aqui compensaria) : " << (long)std::ceil(ny_critico) << " filas" << std::endl;
        } else {
            std::cout << "  NY critico : no existe -> ni con reparto perfecto (overhead=0) el kernel" << std::endl;
            std::cout << "    heterogeneo llega a la velocidad de la GPU pura. La CPU es demasiado" << std::endl;
            std::cout << "    lenta respecto a la GPU en este hardware/precision para compensar el" << std::endl;
            std::cout << "    coste extra del kernel heterogeneo, sea cual sea el tamano del problema." << std::endl;
        }

        // NOTA (bug corregido): comparar T_predicted_het*SAFETY_FACTOR < T_baseline
        // es matematicamente incorrecto, porque T_predicted_het tiene un termino
        // que crece con NY (la pendiente "denom" por fila) y multiplicar TODO
        // el tiempo previsto por SAFETY_FACTOR tambien infla esa pendiente. Si el
        // margen bruto por fila (denom) es pequeño, SAFETY_FACTOR*denom puede
        // superar a t_gpu_pure y la condicion se vuelve imposible de cumplir para
        // CUALQUIER NY, por grande que sea -> nunca se elegiria GPU+CPU aunque el
        // problema fuera arbitrariamente grande, lo cual no tiene sentido fisico.
        //
        // La comparacion correcta es ADITIVA: el margen total que se gana
        // repartiendo (NY * denom) debe superar el overhead fijo del paso
        // heterogeneo, con un factor de seguridad aplicado solo al overhead
        // (que es la parte de coste fijo, no al termino que escala con NY):
        //
        //     NY * denom > SAFETY_FACTOR * t_ovh
        //
        // Esto si conserva la propiedad esperada: si denom > 0 (el reparto ideal
        // es mas rapido que la GPU sola, fila a fila), existe siempre un NY lo
        // bastante grande que lo compensa.
        double margen_total = (double)NY * denom;
        double umbral_seguridad = SAFETY_FACTOR * t_ovh;
        usar_heterogeneo = (denom > 0.0) && (margen_total > umbral_seguridad);

        if (usar_heterogeneo) {
            int ny_gpu_ideal = (int)std::lround((double)NY * t_cpu / (t_gpu_het + t_cpu));
            ny_gpu_ideal = std::max(NY_GPU_MIN, std::min(NY_GPU_MAX, ny_gpu_ideal));
            std::cout << "  DECISION: usar GPU+CPU. Reparto inicial ny_gpu = " << ny_gpu_ideal
                      << " (margen total " << margen_total * 1000.0 << " us/paso > umbral de seguridad "
                      << umbral_seguridad * 1000.0 << " us/paso)" << std::endl;
            migrate_rows(d_f, h_f, ny_gpu, ny_gpu_ideal);
            ny_gpu = ny_gpu_ideal;
            refresh_halo_after_migration(d_f, h_halo_gpu_pinned, h_f, h_halo_cpu, ny_gpu);
        } else {
            std::cout << "  DECISION: colapsar a GPU sola. El margen total repartiendo ("
                      << margen_total * 1000.0 << " us/paso) no supera el overhead fijo con margen de"
                      << " seguridad (" << umbral_seguridad * 1000.0 << " us/paso, factor " << SAFETY_FACTOR
                      << "x) para NY=" << NY << "." << std::endl;
            migrate_rows(d_f, h_f, ny_gpu, NY);
            ny_gpu = NY;
        }
        std::cout << "=========================================================" << std::endl;
    }

    FILE* log_fp = fopen("rebalanceo_log.csv", "w");
    if (log_fp) fprintf(log_fp, "step,ny_gpu,t_gpu_us_por_fila,t_cpu_us_por_fila\n");

    if (usar_heterogeneo) {
        // --- Resto de la simulacion: bucle heterogeneo con reequilibrio dinamico ---
        dim3 gridSize((NX + blockSize.x - 1) / blockSize.x, (ny_gpu + blockSize.y - 1) / blockSize.y);
        double gpu_ms_window = 0.0, cpu_ms_window = 0.0;
        int window_steps = 0;

        for (int step = 0; step < pasos_restantes; ++step) {
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

            gpu_ms_window += gpu_ms; cpu_ms_window += cpu_ms; window_steps++;

            if (window_steps == REBALANCE_WINDOW) {
                int ny_cpu = NY - ny_gpu;
                double t_gpu_row_ms = gpu_ms_window / window_steps / ny_gpu;
                double t_cpu_row_ms = cpu_ms_window / window_steps / ny_cpu;
                int ideal_ny_gpu = (int)std::lround(NY * t_cpu_row_ms / (t_gpu_row_ms + t_cpu_row_ms));
                ideal_ny_gpu = std::max(NY_GPU_MIN, std::min(NY_GPU_MAX, ideal_ny_gpu));
                int delta = ideal_ny_gpu - ny_gpu;
                delta = std::max(-MAX_STEP_ROWS, std::min(MAX_STEP_ROWS, delta));
                // Banda muerta: un ajuste minusculo no compensa el coste de
                // migrar filas (cudaMemcpy2D + rehacer el halo), asi que se
                // ignora y se espera a que el desequilibrio real se acumule.
                int new_ny_gpu = ny_gpu;
                if (std::abs(delta) >= MIN_STEP_ROWS) new_ny_gpu = ny_gpu + delta;

                if (log_fp) fprintf(log_fp, "%d,%d,%.4f,%.4f\n", step, ny_gpu, t_gpu_row_ms * 1000.0, t_cpu_row_ms * 1000.0);

                if (new_ny_gpu != ny_gpu) {
                    migrate_rows(d_f, h_f, ny_gpu, new_ny_gpu);
                    ny_gpu = new_ny_gpu;
                    gridSize.y = (ny_gpu + blockSize.y - 1) / blockSize.y;
                    refresh_halo_after_migration(d_f, h_halo_gpu_pinned, h_f, h_halo_cpu, ny_gpu);
                    std::cout << "  [reequilibrio] paso " << step << ": ny_gpu -> " << ny_gpu << std::endl;
                }
                gpu_ms_window = 0.0; cpu_ms_window = 0.0; window_steps = 0;
            }
        }

        // Unwrap final heterogeneo
        cudaMemcpyAsync(d_halo_from_cpu, h_halo_cpu.data(), 9 * NX * sizeof(real_t), cudaMemcpyHostToDevice, stream);
        streaming_only_het<<<gridSize, blockSize, 0, stream>>>(d_f, d_f_next, d_obstacle, d_halo_from_cpu, u_inflow, ny_gpu);
        std::vector<real_t> h_halo_gpu_for_last(h_halo_gpu_pinned, h_halo_gpu_pinned + 9 * NX);
        cpu_streaming_only(h_f, h_f_next, h_obs, h_halo_gpu_for_last, ny_gpu);
        cudaStreamSynchronize(stream);
        std::swap(d_f, d_f_next);
        h_f.swap(h_f_next);

    } else {
        // --- Resto de la simulacion: GPU pura, cero llamadas a CPU/halo ---
        dim3 gridSize((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y);
        for (int step = 0; step < pasos_restantes; ++step) {
            collide_stream_fused_kernel<<<gridSize, blockSize>>>(d_f, d_f_next, d_obstacle, tau, u_inflow);
            std::swap(d_f, d_f_next);
        }
        streaming_only_kernel<<<gridSize, blockSize>>>(d_f, d_f_next, d_obstacle, u_inflow);
        std::swap(d_f, d_f_next);
        cudaDeviceSynchronize();
    }

    if (log_fp) fclose(log_fp);

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(end_time - start_time).count();
    double mlups = (double(NX) * NY * NUM_STEPS) / (total_sec * 1e6);
    std::cout << "=========================================================" << std::endl;
    std::cout << "Estrategia final: " << (usar_heterogeneo ? "GPU+CPU" : "GPU sola") << "  (ny_gpu final = " << ny_gpu << " / NY = " << NY << ")" << std::endl;
    std::cout << "Tiempo Total: " << total_sec << " s. Rendimiento: " << mlups << " MLUPS" << std::endl;
    std::cout << "=========================================================" << std::endl;

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
