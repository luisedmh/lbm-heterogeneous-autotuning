#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <cstdlib>

// Dimensiones del dominio
const int NX = 400;
const int NY = 100;
const int NUM_STEPS = 8000; // Número de instancias de tiempo q se van a observar
const int SAVE_EVERY = 100; // Cada cuanto guardamos archivo como imagen

// Parámetros físicos
const double tau = 0.56; // Más cercano a 0.5 = menor viscosidad = más turbulencia
const double u_inflow = 0.1; // Velocidad de entrada del "viento"

// Vectores de dirección D2Q9
const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1}; // Velocidades "horizontales" (eje X)
const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1}; // Velocidades "verticales" (eje Y)
const double w[9] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 
                     1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0}; // Para compensar la diferencia entre moverse diagonalmente y perpendicularmente se usan los pesos
const int noslip[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6}; // Direcciones inversas para simular los rebotes

// --- NUESTRA ESTRUCTURA INTUITIVA ---
struct Cell {
    double f[9];
};

// Función para calcular el equilibrio de una dirección específica
inline double equilibrium(int i, double rho, double ux, double uy) { // $$f_i^{eq} = w_i \rho \left(1 + 3(\vec{c}_i \cdot \vec{u}) + 4.5(\vec{c}_i \cdot \vec{u})^2 - 1.5 u^2\right)$$
    double cu = cx[i] * ux + cy[i] * uy;
    double u2 = ux * ux + uy * uy;
    return w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
}

void save_vtk(int step, const std::vector<Cell>& grid, const std::vector<bool>& obstacle) {
    namespace fs = std::filesystem;
    fs::path out_dir = fs::path(std::getenv("HOME")) / "Documents/Proyectos/Informática/HPC/LMB_Autotuning/results/phase1";
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

int main() {
    // Rejilla actual y rejilla para el siguiente paso de tiempo
    std::vector<Cell> grid(NX * NY);
    std::vector<Cell> next_grid(NX * NY);
    std::vector<bool> obstacle(NX * NY, false);

    // Dibujar el obstáculo (Cilindro)
    int cx_cyl = NX / 4, cy_cyl = NY / 2, r_cyl = NY / 10;
    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            if ((x - cx_cyl)*(x - cx_cyl) + (y - cy_cyl)*(y - cy_cyl) < r_cyl * r_cyl) {
                obstacle[y * NX + x] = true;
            }
        }
    }

    // Inicializar todo el fluido en estado de equilibrio básico
    for (int idx = 0; idx < NX * NY; ++idx) {
        for (int i = 0; i < 9; ++i) {
            grid[idx].f[i] = equilibrium(i, 1.0, u_inflow, 0.0);
        }
    }

    // --- BUCLE PRINCIPAL DE SIMULACIÓN ---
    for (int step = 0; step <= NUM_STEPS; ++step) { // Para cada paso
        
        for (int y = 0; y < NY; ++y) {
            for (int x = 0; x < NX; ++x) { // Para cada celda
                int idx = y * NX + x; // Posición de la matriz representada en línea
                
                if (obstacle[idx]) continue; // Si es sólido, no hay colisión ni flujo dentro

                // PASO 1: Calcular Macroscópicos (Densidad y Velocidad)
                double rho = 0;
                double ux = 0;
                double uy = 0;
                for (int i = 0; i < 9; ++i) {
                    rho += grid[idx].f[i];  // Densidad = Sumatorio de particulas en todas las direcciones
                    ux  += grid[idx].f[i] * cx[i];
                    uy  += grid[idx].f[i] * cy[i];
                }
                ux /= rho;  // Velocidad = 1/rho * Sumatorio de las particulas por sus velocidades
                uy /= rho;

                // PASO 2 Y 3: Colisión y Streaming combinados
                for (int i = 0; i < 9; ++i) {
                    double fi = grid[idx].f[i];
                    double feq = equilibrium(i, rho, ux, uy);
                    
                    // Ecuación de colisión BGK
                    double f_post_collision = fi - (fi - feq) / tau; 

                    // La posicion i de la celda idx a dd se moverá?
                    int next_x = x + cx[i];
                    int next_y = y + cy[i];

                    // PASO 4: Control de fronteras (Rebotes)
                    // Rebote en paredes superior/inferior
                    if (next_y < 0 || next_y >= NY) {
                        next_grid[idx].f[noslip[i]] = f_post_collision; // Recordar noslip es la direccion contraria, rebote
                    }
                    // Rebote contra el cilindro
                    else if (obstacle[next_y * NX + next_x]) {
                        next_grid[idx].f[noslip[i]] = f_post_collision;
                    }
                    // Movimiento libre normal en el interior del fluido
                    else if (next_x >= 0 && next_x < NX) {
                        int next_idx = next_y * NX + next_x;
                        next_grid[next_idx].f[i] = f_post_collision;
                    }
                }
            }
        }

        // CONDICIONES DE CONTORNO DE ENTRADA Y SALIDA
        for (int y = 0; y < NY; ++y) {
            // Entrada (Izquierda): Forzamos flujo constante constante
            int idx_in = y * NX + 0;
            for (int i = 0; i < 9; ++i) {
                next_grid[idx_in].f[i] = equilibrium(i, 1.0, u_inflow, 0.0);
            }
            // Salida (Derecha): Copia amortiguada de la columna anterior
            int idx_out = y * NX + (NX - 1);
            int idx_prev = y * NX + (NX - 2);
            for (int i = 0; i < 9; ++i) {
                next_grid[idx_out].f[i] = next_grid[idx_prev].f[i];
            }
        }

        // El futuro se convierte en el presente
        grid = next_grid;

        if (step % SAVE_EVERY == 0) {
            std::cout << "Simulado paso: " << step << " / " << NUM_STEPS << std::endl;
            save_vtk(step, grid, obstacle);
        }
    }

    std::cout << "¡Simulación completada con éxito!" << std::endl;
    return 0;
}