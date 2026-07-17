#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <omp.h> // <--- NUEVO: Biblioteca estándar de OpenMP

const int NX = 400;
const int NY = 100;
const int NUM_STEPS = 8000;
const int SAVE_EVERY = 100;

const double tau = 0.56; 
const double u_inflow = 0.1;

const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
const double w[9] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 
                     1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};
const int noslip[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6}; 

struct Cell {
    double f[9];
};

inline double equilibrium(int i, double rho, double ux, double uy) {
    double cu = cx[i] * ux + cy[i] * uy;
    double u2 = ux * ux + uy * uy;
    return w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
}

void save_vtk(int step, const std::vector<Cell>& grid, const std::vector<bool>& obstacle);

int main() {
    std::vector<Cell> grid(NX * NY);
    std::vector<Cell> next_grid(NX * NY);
    std::vector<bool> obstacle(NX * NY, false);

    // Obstáculo (Cilindro)
    int cx_cyl = NX / 4, cy_cyl = NY / 2, r_cyl = NY / 10;
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            if ((x - cx_cyl)*(x - cx_cyl) + (y - cy_cyl)*(y - cy_cyl) < r_cyl * r_cyl) {
                obstacle[y * NX + x] = true;
            }
        }
    }

    // Inicializar al equilibrio
    for (int idx = 0; idx < NX * NY; ++idx) {
        for (int i = 0; i < 9; ++i) {
            grid[idx].f[i] = equilibrium(i, 1.0, u_inflow, 0.0);
        }
    }

    std::cout << "Iniciando simulación paralela en CPU..." << std::endl;
    
    // --- MEDIDOR DE TIEMPO HPC ---
    double start_time = omp_get_wtime();

    // --- BUCLE PRINCIPAL DE SIMULACIÓN ---
    for (int step = 0; step <= NUM_STEPS; ++step) {
        
        // PASO 1: COLISIÓN LOCAL PARALELA
        // collapse(2) fusiona los bucles de 'y' y 'x' en un único bucle gigante de 40,000 iteraciones,
        // permitiendo que OpenMP reparta el trabajo perfectamente entre tus 12 hilos.
        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) {
                int idx = y * NX + x;
                if (obstacle[idx]) continue;

                double rho = 0, ux = 0, uy = 0;
                for (int i = 0; i < 9; ++i) {
                    rho += grid[idx].f[i];
                    ux  += grid[idx].f[i] * cx[i];
                    uy  += grid[idx].f[i] * cy[i];
                }
                ux /= rho;
                uy /= rho;

                for (int i = 0; i < 9; ++i) {
                    double feq = equilibrium(i, rho, ux, uy);
                    grid[idx].f[i] = grid[idx].f[i] - (grid[idx].f[i] - feq) / tau;
                }
            }
        }

        // PASO 2: STREAMING TIPO "PULL" PARALELO
        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < NY; ++y) {
            for (int x = 1; x < NX - 1; ++x) {
                int idx = y * NX + x;
                if (obstacle[idx]) continue;

                for (int i = 0; i < 9; ++i) {
                    int prev_x = x - cx[i];
                    int prev_y = y - cy[i];
                    int prev_idx = prev_y * NX + prev_x;

                    if (prev_y < 0 || prev_y >= NY) {
                        next_grid[idx].f[i] = grid[idx].f[noslip[i]];
                    }
                    else if (obstacle[prev_idx]) {
                        next_grid[idx].f[i] = grid[idx].f[noslip[i]];
                    }
                    else {
                        next_grid[idx].f[i] = grid[prev_idx].f[i];
                    }
                }
            }
        }

        // PASO 3: CONDICIONES DE CONTORNO PARALELAS (Fronteras Izquierda y Derecha)
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < NY; ++y) {
            int idx_in = y * NX + 0;
            for (int i = 0; i < 9; ++i) {
                next_grid[idx_in].f[i] = equilibrium(i, 1.0, u_inflow, 0.0);
            }
            int idx_out = y * NX + (NX - 1);
            int idx_prev = y * NX + (NX - 2);
            for (int i = 0; i < 9; ++i) {
                next_grid[idx_out].f[i] = next_grid[idx_prev].f[i];
            }
        }

        // OPTIMIZACIÓN DE ALTO RENDIMIENTO: Pointer Swap
        // En lugar de hacer 'grid = next_grid' (que copia 2.8 MB de memoria en cada paso),
        // intercambiamos los punteros internos de los vectores de forma instantánea O(1).
        std::swap(grid, next_grid);

        if (step % SAVE_EVERY == 0) {
            std::cout << "Simulado paso: " << step << " / " << NUM_STEPS << std::endl;
            // Nota: No paralelizar la escritura en disco; debe ser secuencial para no corromper el archivo.
            save_vtk(step, grid, obstacle);
        }
    }

    double end_time = omp_get_wtime();
    std::cout << "=========================================================" << std::endl;
    std::cout << "¡Simulación completada con éxito!" << std::endl;
    std::cout << "Tiempo de cómputo: " << (end_time - start_time) << " segundos." << std::endl;
    std::cout << "=========================================================" << std::endl;
    return 0;
}

// (La función save_vtk se mantiene idéntica para asegurar compatibilidad con ParaView)
void save_vtk(int step, const std::vector<Cell>& grid, const std::vector<bool>& obstacle) {
    namespace fs = std::filesystem;
    fs::path out_dir = fs::path(std::getenv("HOME")) / "Documents/Proyectos/Informática/HPC/LMB_Autotuning/results/phase2";
    fs::create_directories(out_dir);
    std::stringstream ss;
    ss << "fluid_" << std::setw(4) << std::setfill('0') << step << ".vtk";
    std::ofstream out((out_dir / ss.str()).string());

    out << "# vtk DataFile Version 3.0\nLBM 2D9Q\nASCII\nDATASET STRUCTURED_POINTS\n";
    out << "DIMENSIONS " << NX << " " << NY << " 1\nORIGIN 0 0 0\nSPACING 1 1 1\n";
    out << "POINT_DATA " << NX * NY << "\nSCALARS velocity_magnitude double 1\nLOOKUP_TABLE default\n";

    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = y * NX + x;
            if (obstacle[idx]) {
                out << 0.0 << "\n";
            } else {
                double rho = 0, ux = 0, uy = 0;
                for (int i = 0; i < 9; ++i) {
                    rho += grid[idx].f[i];
                    ux  += grid[idx].f[i] * cx[i];
                    uy  += grid[idx].f[i] * cy[i];
                }
                out << std::sqrt((ux/rho)*(ux/rho) + (uy/rho)*(uy/rho)) << "\n";
            }
        }
    }

    out << "VECTORS velocity double\n";
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = y * NX + x;
            if (obstacle[idx]) {
                out << "0.0 0.0 0.0\n";
            } else {
                double rho = 0, ux = 0, uy = 0;
                for (int i = 0; i < 9; ++i) {
                    rho += grid[idx].f[i];
                    ux  += grid[idx].f[i] * cx[i];
                    uy  += grid[idx].f[i] * cy[i];
                }
                out << ux/rho << " " << uy/rho << " 0.0\n";
            }
        }
    }
}