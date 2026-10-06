# LBM D2Q9 heterogéneo CPU+GPU — de comparar con FluidX3D a un reequilibrador dinámico y auto-calibrado

Este directorio recoge, en orden cronológico, cada etapa del trabajo: desde el
código solo-GPU que se comparó con el estándar de la industria (FluidX3D)
hasta la versión final, que decide por sí misma cómo repartir el trabajo
entre CPU y GPU. Cada subcarpeta corresponde a un hito con validación propia
— nada de lo que hay aquí se dio por bueno sin comparar sus resultados,
celda a celda, contra una referencia ya verificada.

## Resumen de una frase

Un simulador LBM D2Q9 en CUDA que primero se validó y comparó de forma justa
contra FluidX3D (superándolo un ~39% en FP32, mismo tamaño de problema, misma
precisión), y después se extendió con reparto de trabajo CPU+GPU que se
reequilibra solo en tiempo real y que decide midiendo — no asumiendo — si
merece la pena usar la CPU para un problema y una precisión dados,
consiguiendo entre +24.7% y +31.3% sobre GPU sola en FP64.

## Índice de carpetas

| Carpeta | Qué contiene | Resultado clave |
|---|---|---|
| `01_FluidX3D_referencia/` | Configuración usada de FluidX3D (el "estándar de la industria" con el que se comparó) | Punto de referencia externo |
| `02_solo_GPU/` | Código LBM propio, solo GPU, versión original y versión fusionada | 7341 MLUPS, +39% sobre FluidX3D en igualdad de condiciones |
| `03_heterogeneo_v2/` | Primera versión heterogénea CPU+GPU correcta (reparto fijo) | +11% (FP64) sobre GPU sola, con reparto sin optimizar |
| `04_heterogeneo_dinamico/` | Reequilibrio automático del reparto durante la ejecución | +24.7% (FP64), converge solo sin conocer el óptimo de antemano |
| `05_heterogeneo_auto/` | Versión final: decide sola si usar CPU+GPU o GPU sola, midiendo | +31.3% (FP64) cuando compensa; se autodesactiva sin penalización cuando no |

## La historia completa, paso a paso

### 1. El estándar de la industria (`01_FluidX3D_referencia/`)

Antes de afirmar que un código es rápido hace falta algo con qué compararlo.
Se identificaron dos referencias del campo: **FluidX3D** (probablemente el
LBM en GPU más rápido disponible públicamente) y **OpenLB** (framework
académico, usado más adelante como referencia de balanceo de carga). Se
instaló FluidX3D y se configuró para una comparación limpia: D2Q9 (2D, igual
que el código propio), sin la compresión `FP16S` (que le daría ventaja
artificial), con un tamaño de dominio (≈991.000 celdas) equivalente al
propio (1.000.000 de celdas). El primer intento de comparación fue inválido
(FluidX3D corría en 3D con D3Q19 y compresión activada, dos simulaciones
distintas que no se podían comparar) — los archivos aquí son la
configuración ya corregida y justa.

**Archivos:**
- `defines.hpp` — activa D2Q9, desactiva `FP16S` y `BENCHMARK`, activa
  `EQUILIBRIUM_BOUNDARIES` (necesaria para las condiciones de entrada/salida
  del caso de prueba).
- `setup.cpp` — el caso de prueba usado: una calle de vórtices de Kármán
  (flujo detrás de un cilindro), con el tamaño de rejilla (`R=44`) ajustado
  para igualar el número de celdas del código propio, y `lbm.run(3000u)`
  para que el número de pasos también coincida.

### 2. Solo GPU: la base que ganó a FluidX3D (`02_solo_GPU/`)

El código propio, solo GPU, pasó por una corrección importante antes de que
el resultado fuera fiable: la primera versión "fusionada" (colisión +
streaming en un único kernel, para eliminar una pasada completa de memoria)
tenía un bug que solo aparecía cerca del obstáculo — el orden de aplicar la
física (relajar-luego-mover vs. mover-luego-relajar) no coincidía con la
versión original en el primer paso de la simulación, y eso desajustaba la
condición de rebote en la pared del cilindro. Se detectó comparando ambas
versiones a los 5 pasos, se corrigió, y se revalidó hasta una diferencia de
`2.220e-16` — el límite de precisión de un `double`, es decir, ya no hay
ningún error, solo el ruido de redondeo inevitable de reordenar operaciones
en coma flotante.

Con esa versión corregida y validada, la comparación final con FluidX3D en
FP32 puro (sin compresión, mismo tamaño de problema) dio **7341 MLUPS del
código propio frente a 5272 de FluidX3D — un 39% por delante**, en un
resultado legítimo y reproducible.

**Archivos:**
- `lbm_gpu_original_con_dump.cu` — la versión original (colisión y streaming
  como dos kernels separados), usada como referencia de validación bit a bit
  para todas las versiones fusionadas posteriores.
- `lbm_gpu_fused.cu` — la versión fusionada corregida y validada (la que dio
  7341 MLUPS). Es también la base de la que parte todo el código heterogéneo
  posterior: sus tres kernels (`collision_only_kernel`, `streaming_only_kernel`,
  `collide_stream_fused_kernel`) se reutilizan sin cambios en las versiones
  GPU-sola de las carpetas siguientes.
- `compara_resultados.py` — script de validación: compara dos volcados
  binarios del estado final de la simulación celda a celda y calcula la
  diferencia máxima y media. Un criterio de `<1e-12` se considera "física y
  numéricamente correcto" (ruido de redondeo, no un bug).

### 3. Primera versión heterogénea correcta (`03_heterogeneo_v2/`)

El código heterogéneo CPU+GPU original de Luis (el que daba el 1.2x inicial)
tenía dos bugs de fondo, encontrados por revisión de código antes de tocar
nada del reparto:

1. **Condición de carrera en la CPU**: colisionaba y hacía streaming de la
   misma celda dentro del mismo bucle paralelo de OpenMP, sin garantía de que
   una celda vecina ya estuviera colisionada cuando otro hilo la leía.
2. **Halo unidireccional**: la franja de contacto entre la parte de GPU y la
   de CPU solo mandaba datos GPU→CPU, nunca al revés — la GPU trataba esa
   frontera como una pared en vez de como una frontera real con la CPU.

`lbm_heterogeneo_v2.cu` aplica a los dos lados (GPU y CPU) el mismo patrón
"recoger y relajar" que ya usa el kernel fusionado solo-GPU, lo que resuelve
la condición de carrera de raíz y hace natural el intercambio de halo en los
dos sentidos. Con un reparto fijo (900 filas GPU / 100 filas CPU), validado a
`5.718e-15` de diferencia, dio **+11% sobre GPU sola en FP64**. En FP32, ese
mismo reparto fijo resultó contraproducente (2936 MLUPS frente a 7332 de GPU
sola) — la primera señal de que un reparto fijo no vale para todas las
precisiones, lo que motivó el siguiente paso.

*(El código heterogéneo original, con los dos bugs, no se incluye como
archivo aparte: solo existió como código pegado en la conversación, nunca se
guardó como fichero independiente, y queda completamente superado por esta
versión corregida.)*

**Archivos:**
- `lbm_heterogeneo_v2.cu` — reparto GPU/CPU fijo (`NY_GPU_INIT`, ajustable a
  mano), con halo bidireccional correcto.

### 4. Reequilibrio dinámico (`04_heterogeneo_dinamico/`)

Antes de optimizar el reparto fijo a mano, se midió la CPU y la GPU por
separado en las dos precisiones, lo que reveló el concepto que dominaría el
resto del proyecto: existe un **overhead fijo por paso** (lanzar el kernel,
copiar el halo en los dos sentidos, sincronizar, el fork-join de OpenMP) que
no depende de cuántas filas se repartan, solo de que se reparta en absoluto.
En FP32, donde la GPU es ~13x más rápida que en FP64 pero la CPU apenas
mejora, ese overhead se come cualquier margen posible — ni siquiera el
reparto teóricamente óptimo (962/38) conseguía batir a la GPU sola.

Se buscó en la literatura si alguien ya reequilibraba la carga CPU/GPU **en
tiempo real** durante la simulación (no solo una vez, al principio). No se
encontró: OpenLB calcula el reparto una vez con un algoritmo genético y lo
deja fijo; Calore et al. (2017) hacen auto-tuning con mini-benchmarks al
principio, también fijo el resto de la ejecución; WaLBerla particiona de
forma estática. `lbm_heterogeneo_dinamico.cu` reequilibra el reparto cada
`REBALANCE_WINDOW` pasos, midiendo en vivo cuánto tarda cada lado por fila y
migrando filas entre GPU y CPU cuando el desequilibrio lo justifica.
Arrancado deliberadamente lejos del óptimo (`ny_gpu=500` en vez de ~700), el
sistema convergió solo hasta estabilizarse entre 660-700, dando **709.2
MLUPS frente a 568.8 de GPU sola (+24.7%) en FP64**, validado a `9.381e-15`.

**Archivos:**
- `lbm_heterogeneo_dinamico.cu` — reequilibrio dinámico por ventanas, sin
  ninguna decisión automática de si usar la CPU o no (eso llega en el
  siguiente paso).
- `rebalanceo_log.csv` — registro paso a paso del reparto durante una
  ejecución (columnas: paso, `ny_gpu`, tiempo de GPU y CPU por fila). Útil
  para graficar la convergencia.

### 5. Auto-calibración: que el programa decida solo (`05_heterogeneo_auto/`)

La versión final generaliza la idea: en vez de asumir "FP64 sí, FP32 no", el
programa mide sus propios tiempos (GPU y CPU por fila, overhead fijo) al
arrancar y decide él solo, para cualquier tamaño de problema y cualquier
precisión, si compensa repartir o no. Si no compensa, colapsa a GPU sola sin
pagar ni un microsegundo de coste heterogéneo residual.

Llegar a esta versión llevó varias rondas de depuración real, documentadas
con detalle en `RESULTADOS_HETEROGENEO.md` (dentro de esta misma carpeta):
una regresión real del 17% causada por comparar el kernel heterogéneo
consigo mismo en vez de contra un kernel GPU puro medido aparte; una fórmula
de decisión con un factor de seguridad matemáticamente roto que hacía
imposible activar el reparto en FP32 por fino que fuera el margen; ausencia
total de comprobación de errores de CUDA, que en un caso de memoria de GPU
insuficiente dio un resultado 10 veces inflado sin ningún aviso; y, por
último, el hallazgo de que el coste de "hablar con la CPU" en FP32 depende
del estado térmico de la GPU (varía entre 2.5% y 8.5% según el momento),
lo que explica por qué, en esta GPU concreta, no se encontró ningún tamaño
de problema donde el reparto CPU+GPU compensara de forma fiable en FP32 —
y por qué el sistema, correctamente, nunca lo activó por error.

**Archivos:**
- `lbm_heterogeneo_auto.cu` — versión final: auto-calibración (mide GPU pura,
  GPU heterogénea y CPU por separado), decisión GPU-sola vs. GPU+CPU con
  comprobación de errores de CUDA, y reequilibrio dinámico del reparto para
  el resto de la simulación si decide usar la CPU.
- `RESULTADOS_HETEROGENEO.md` — informe con todos los resultados numéricos
  (tablas FP64/FP32, validaciones), la lista completa de bugs encontrados y
  corregidos durante el desarrollo, y la comparación con OpenLB/Calore et
  al./WaLBerla.

## Cómo se validó cada paso

En todos los casos, el protocolo fue el mismo: ejecutar una prueba corta (5
pasos) para detectar bugs rápido, y una prueba larga (3000 pasos) para
confirmar que el resultado se mantiene correcto a lo largo de toda la
simulación — comparando siempre, celda a celda, contra `lbm_gpu_fused.cu` en
modo `double` (FP64), con `compara_resultados.py`. Una diferencia del orden
de `1e-13` a `1e-15` se considera validación correcta (ruido de redondeo de
máquina); cualquier cosa por encima de `1e-12` se investigó como un bug real
antes de aceptar ningún número de rendimiento.