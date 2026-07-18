# Diseño: Selección de fase en el Makefile

## Contexto

El proyecto organiza el código por fases (`src/Phase1_Sequential/`, `src/Phase2_OpenMP/`, futuras fases). El Makefile actual compila recursivamente **todos** los `.cc`/`.cu` de `src/` en un único binario `bin/lbm`. Esto deja de funcionar en cuanto exista más de una fase con su propio `main()` (colisión de símbolos al enlazar).

## Objetivo

Permitir elegir qué fase compilar (`make build`), ejecutar (`make run`) y limpiar (`make clean`), sin tener que editar el Makefile al añadir una fase nueva.

## Detección de fases

Las fases se detectan automáticamente como las subcarpetas directas de `src/` (ej. `Phase1_Sequential`, `Phase2_OpenMP`). No requiere mantenimiento manual del Makefile.

## Selección de fase

- `make build`, `make run`, `make clean` sin argumentos muestran un menú numerado (fases en orden alfabético) y leen la elección por teclado (`read`).
- Se puede saltar el menú con `PHASE=<valor>`, donde `<valor>` es **el nombre exacto de la carpeta** (`PHASE=Phase1_Sequential`) **o el número de orden en el menú** (`PHASE=1`).
- `make clean` añade una opción extra "Todas las fases" en el menú (y acepta `PHASE=ALL` como atajo) que borra `build/` y `bin/` completos.
- Internamente, el target visible (`build`/`run`/`clean`) resuelve la fase en su recipe y relanza `make` recursivamente con `PHASE=<fase-resuelta>` fijo (target interno `_build`/`_run`/`_clean`), de forma que toda la lógica de compilación (fuentes, objetos, flags de CUDA) se evalúa ya con una única fase conocida.

## Artefactos por fase

- Objetos: `build/<Fase>/*.o` (ya sigue este patrón).
- Binario: `bin/lbm_<Fase>` (ej. `bin/lbm_Phase1_Sequential`). Cada fase tiene su propio binario; no se sobrescriben entre sí.

## Detección de CUDA

Pasa a evaluarse por fase: se buscan `.cu` dentro de `src/<Fase>/` (antes se buscaba en todo `src/` recursivamente). Si hay `.cu` en la fase y `nvcc` está disponible, se activa `-DUSE_CUDA` y se linkea el runtime de CUDA solo para esa fase.

## `make info`

Se actualiza para listar las fases detectadas (no depende de `PHASE`).

## Fuera de alcance

- No se toca la lógica de detección de `include/` ni las flags de compilación (`-O3 -Wall -Wextra -fopenmp`, etc.).
- No se migra/renombra el binario `bin/lbm` ni el objeto `build/Phase1_Sequential/main.o` ya existentes en el árbol de trabajo; quedan obsoletos y se limpian con `make clean` cuando el usuario lo decida.
- No se añade CI ni tests automatizados del Makefile (proyecto sin infraestructura de CI existente).