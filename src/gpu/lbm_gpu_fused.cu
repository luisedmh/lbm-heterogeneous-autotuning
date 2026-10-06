// ============================================================================
// LBM D2Q9 - SOLO GPU - VERSION "FUSIONADA" (colisión+streaming en 1 kernel)
// v2 - corrige un bug de la v1 en el primer y ultimo paso (ver mas abajo)
// ============================================================================
//
// IDEA GENERAL: tu código original hace, en cada paso, dos kernels:
//   1) collision_kernel: relaja (colisiona) cada celda EN SU SITIO.
//   2) streaming_kernel: mueve (streaming) esos valores ya relajados hacia
//      las celdas vecinas.
// Eso son dos pasadas completas de lectura+escritura sobre los 9 arrays f
// por celda y por paso. Como LBM está limitado por ancho de banda de
// memoria (no por cuánto calcula la GPU, que le sobra), fusionar ambas
// operaciones en un único kernel por paso es la optimización de mayor
// impacto posible antes de tocar la precisión.
//
// EL BUG DE LA VERSION ANTERIOR (por si te lo preguntas o lo explicas en la
// memoria): para fusionar sin condiciones de carrera, cada kernel tiene que
// leer datos que YA fueron relajados en un paso ANTERIOR (nunca "a medio
// calcular" por otro hilo en el mismo lanzamiento). Eso significa que, en
// vez de operar sobre el dato físico bruto, el buffer que se pasa de un
// kernel a otro tiene que contener SIEMPRE "resultado de una colisión", no
// "dato físico sin colisionar". El problema es el primerísimo paso: el
// array que inicializas (equilibrio de u_inflow) es un dato físico bruto,
// no el resultado de ninguna colisión previa. Aplicarle directamente la
// regla de "recoger de los vecinos con rebote" (como hace el kernel
// fusionado) mezcla direcciones de forma prematura en las celdas pegadas al
// obstáculo, un paso antes de lo que tocaría. Por eso el error aparecía ya
// en el paso 1 y quedaba pegado al cilindro.
//
// LA SOLUCIÓN: dejar el primer paso como una colisión "suelta" (igual que
// tu collision_kernel original, sin streaming) y el último paso como un
// streaming "suelto" (igual que tu streaming_kernel original, sin
// colisión). Todos los pasos intermedios sí se fusionan. Para tus 3000
// pasos, eso es 1 + 2999 + 1 = 3001 lanzamientos de kernel en vez de los
// 6000 de tu versión original — casi toda la ganancia, y ahora sí correcto.
//
// Precisión conmutable en tiempo de compilación (real_t = double o float):
//   nvcc -O3 -arch=native lbm_gpu_fused.cu -o lbm_fused_fp64                      (validacion)
//   nvcc -O3 -arch=native -DUSE_SINGLE_PRECISION lbm_gpu_fused.cu -o lbm_fused_fp32  (velocidad)
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

const int NX = 1000;
const int NY = 1000;
const int NUM_STEPS = 3000;

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
// KERNEL A: colisión sola (idéntica a tu collision_kernel original).
// Se usa UNA sola vez, al principio, para preparar el terreno para el
// kernel fusionado.
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
        rho += fi;
        ux  += fi * d_cx[i];
        uy  += fi * d_cy[i];
    }
    ux /= rho;
    uy /= rho;
    real_t u2 = ux * ux + uy * uy;

    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        real_t cu = d_cx[i] * ux + d_cy[i] * uy;
        real_t feq = d_w[i] * rho * (REAL_LIT(1.0) + REAL_LIT(3.0) * cu + REAL_LIT(4.5) * cu * cu - REAL_LIT(1.5) * u2);
        int f_idx = i * N + idx;
        f[f_idx] = f[f_idx] - (f[f_idx] - feq) / tau_param;
    }
}

// ============================================================================
// KERNEL B: streaming solo (idéntico a tu streaming_kernel original, sin
// ninguna colisión). Se usa UNA sola vez, al final, para "desenvolver" el
// último resultado colisionado en el estado físico final que se compara /
// vuelca a disco.
// ============================================================================
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

// ============================================================================
// KERNEL C: colisión+streaming fusionados. Se usa para TODOS los pasos
// intermedios (todos salvo el primero y el último). Recibe un buffer que ya
// es "resultado de una colisión" y devuelve otro que también lo es.
// ============================================================================
__global__ void collide_stream_fused_kernel(const real_t* __restrict__ f, real_t* __restrict__ f_next,
                                             const bool* __restrict__ obstacle,
                                             real_t tau_param, real_t u_inflow_param) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;
    int idx = y * NX + x;
    const int N = NX * NY;

    // --- Frontera de entrada: en equilibrio siempre, colisionar no la cambia ---
    if (x == 0) {
        real_t u2 = u_inflow_param * u_inflow_param;
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            real_t cu = d_cx[i] * u_inflow_param;
            f_next[i * N + idx] = d_w[i] * (REAL_LIT(1.0) + REAL_LIT(3.0) * cu + REAL_LIT(4.5) * cu * cu - REAL_LIT(1.5) * u2);
        }
        return;
    }

    // --- Frontera de salida: extrapolar Y LUEGO colisionar el resultado ---
    // (a diferencia de la v1, aquí SÍ se relaja el valor extrapolado, para
    // que la columna NX-2 reciba de esta columna un dato "post-colisión"
    // coherente con el resto del dominio, igual que en tu código original)
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
        ux /= rho;
        uy /= rho;
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

    // --- Interior: recolectar (streaming) desde el buffer congelado y colisionar ---
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
    ux /= rho;
    uy /= rho;
    real_t u2 = ux * ux + uy * uy;
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        real_t cu = d_cx[i] * ux + d_cy[i] * uy;
        real_t feq = d_w[i] * rho * (REAL_LIT(1.0) + REAL_LIT(3.0) * cu + REAL_LIT(4.5) * cu * cu - REAL_LIT(1.5) * u2);
        f_next[i * N + idx] = g[i] - (g[i] - feq) / tau_param;
    }
}

int main() {
    std::vector<real_t> h_f(9 * NX * NY);
    std::vector<char> h_obs(NX * NY, 0);

    int cx_cyl = NX / 4, cy_cyl = NY / 2, r_cyl = NY / 10;
    for (int y = 0; y < NY; ++y)
        for (int x = 0; x < NX; ++x)
            if ((x - cx_cyl)*(x - cx_cyl) + (y - cy_cyl)*(y - cy_cyl) < r_cyl * r_cyl)
                h_obs[y * NX + x] = 1;

    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = y * NX + x;
            real_t u2_init = u_inflow * u_inflow;
            for (int i = 0; i < 9; ++i) {
                real_t cu = cx[i] * u_inflow;
                h_f[i * NX * NY + idx] = w[i] * (REAL_LIT(1.0) + REAL_LIT(3.0) * cu + REAL_LIT(4.5) * cu * cu - REAL_LIT(1.5) * u2_init);
            }
        }
    }

    real_t *d_f, *d_f_next;
    bool *d_obstacle;
    cudaMalloc(&d_f, 9 * NX * NY * sizeof(real_t));
    cudaMalloc(&d_f_next, 9 * NX * NY * sizeof(real_t));
    cudaMalloc(&d_obstacle, NX * NY * sizeof(bool));

    std::vector<char> h_obs_bool(NX * NY);
    for (int i = 0; i < NX * NY; ++i) h_obs_bool[i] = (h_obs[i] == 1);

    cudaMemcpy(d_f, h_f.data(), 9 * NX * NY * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obstacle, h_obs_bool.data(), NX * NY * sizeof(bool), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(d_cx, cx, 9 * sizeof(int));
    cudaMemcpyToSymbol(d_cy, cy, 9 * sizeof(int));
    cudaMemcpyToSymbol(d_w, w, 9 * sizeof(real_t));
    cudaMemcpyToSymbol(d_noslip, noslip, 9 * sizeof(int));

    dim3 blockSize(32, 8);
    dim3 gridSize((NX + blockSize.x - 1) / blockSize.x, (NY + blockSize.y - 1) / blockSize.y);

#ifdef USE_SINGLE_PRECISION
    std::cout << "Precision: FP32 (float)" << std::endl;
#else
    std::cout << "Precision: FP64 (double) - version de VALIDACION" << std::endl;
#endif
    std::cout << "=========================================================" << std::endl;
    std::cout << ">>> LBM D2Q9 GPU - KERNEL FUSIONADO v2 (colision+streaming)" << std::endl;
    std::cout << "=========================================================" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Paso 0: colision sola (bootstrap)
    collision_only_kernel<<<gridSize, blockSize>>>(d_f, d_obstacle, tau);

    // Pasos 1 .. NUM_STEPS-2: fusionados
    for (int step = 0; step < NUM_STEPS - 1; ++step) {
        collide_stream_fused_kernel<<<gridSize, blockSize>>>(d_f, d_f_next, d_obstacle, tau, u_inflow);
        std::swap(d_f, d_f_next);
    }

    // Paso NUM_STEPS-1: streaming solo (desenvolver al estado fisico final)
    streaming_only_kernel<<<gridSize, blockSize>>>(d_f, d_f_next, d_obstacle, u_inflow);
    std::swap(d_f, d_f_next);

    cudaDeviceSynchronize();

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(end_time - start_time).count();
    double mlups = (double(NX) * NY * NUM_STEPS) / (total_sec * 1e6);

    std::cout << "Tiempo Total: " << total_sec << " segundos." << std::endl;
    std::cout << "Rendimiento : " << mlups << " MLUPS" << std::endl;
    std::cout << "=========================================================" << std::endl;

    {
        std::vector<real_t> h_f_final(9 * NX * NY);
        cudaMemcpy(h_f_final.data(), d_f, 9 * NX * NY * sizeof(real_t), cudaMemcpyDeviceToHost);
        FILE* fp = fopen("f_final.bin", "wb");
        if (fp) {
            fwrite(h_f_final.data(), sizeof(real_t), h_f_final.size(), fp);
            fclose(fp);
            std::cout << "Estado final volcado en f_final.bin (" << h_f_final.size() * sizeof(real_t) << " bytes)" << std::endl;
        }
    }

    cudaFree(d_f); cudaFree(d_f_next); cudaFree(d_obstacle);
    return 0;
}
