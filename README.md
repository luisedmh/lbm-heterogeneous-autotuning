# lbm-heterogeneous-autotuning

Solver LBM (Lattice Boltzmann) 2D para CFD con reparto dinámico de trabajo entre CPU y GPU.
Proyecto de verano.

## Motivación

La mayoría de trabajos sobre autotuning en cómputo tipo stencil deciden el reparto CPU/GPU
(o la precisión numérica) una única vez, al principio de la ejecución. Este proyecto explora
qué pasa si esa decisión se reajusta **mientras la simulación corre**, usando medidas reales
de rendimiento en lugar de una configuración fija.

## Estado actual

| Fase | Contenido | Estado |
|------|-----------|--------|
| 1 | Solver LBM D2Q9 funcional en CPU (OpenMP) y GPU (CUDA), reparto fijo | En progreso |
| 2 | Bucle de reajuste dinámico del reparto CPU/GPU en tiempo de ejecución | Pendiente |
| 3 | Precisión adaptativa por zonas (FP16/FP32) | Exploratorio / trabajo futuro |

## Estructura del repositorio
 
```
.
├── include/lbm/       Cabeceras públicas (grid, solver)
├── src/common/        Código compartido (gestión de la malla)
├── src/cpu/           Implementación CPU (OpenMP)
├── src/cuda/          Implementación GPU (CUDA), se compila solo si nvcc está disponible
├── src/main.c         Punto de entrada y CLI
├── scripts/           Utilidades en Python (procesar y graficar benchmarks)
├── benchmarks/results/ Resultados de benchmarks (CSV, gráficas)
└── docs/notes.md       Notas de estado del arte y referencias
```

## Compilar y ejecutar

Requiere `gcc` con soporte OpenMP. `nvcc` es opcional: si no está disponible, el proyecto
se compila igualmente y solo queda activo el backend de CPU.

```bash
make            # compila (detecta automáticamente si hay CUDA disponible)
make run        # compila y ejecuta con parámetros por defecto
make clean      # limpia binarios y objetos
```

## Añadir código nuevo

El `Makefile` recoge automáticamente **cualquier** archivo `.cc` o `.cu` que añadas bajo `src/`,
en cualquier subcarpeta. No hace falta editar el `Makefile` para que se compile — solo
asegúrate de que el `#include` de tu cabecera use la ruta desde `include/`.

## Referencias de partida

Ver [`docs/notes.md`](docs/notes.md) para el estado del arte y los términos de búsqueda
usados para llegar a él.

## Licencia

MIT — ver [`LICENSE`](LICENSE).
