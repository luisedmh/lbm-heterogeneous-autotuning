# LBM D2Q9 heterogéneo CPU+GPU con reequilibrio dinámico y auto-calibración

## Resumen ejecutivo

Se desarrolló un simulador LBM D2Q9 (colisión BGK + streaming fusionados en un
único kernel CUDA) capaz de repartir el dominio entre GPU y CPU en tiempo de
ejecución, reequilibrando la frontera del reparto cada pocos pasos según
tiempos medidos en vivo, y capaz de decidir por sí mismo — midiendo, no
adivinando — si merece la pena usar la CPU en absoluto para un problema y
precisión dados.

El resultado principal, validado bit a bit contra una implementación GPU-sola
ya verificada: **en FP64, repartir dinámicamente el trabajo entre GPU y CPU da
entre +24.7% y +31.3% de rendimiento sobre GPU sola**, con el sistema
convergiendo automáticamente al reparto óptimo sin conocerlo de antemano. En
FP32, sobre el hardware de pruebas (GPU de 8GB), se comprobó — con
repeticiones controladas, no con una única medida — que el margen real es
negativo o está dominado por ruido térmico/de reloj de la GPU en la mayoría de
condiciones, y el sistema **correctamente decide no usar la CPU en ningún
caso probado**, evitando activamente cualquier regresión. Esa capacidad de
autodetección — "usa la CPU solo cuando de verdad compensa, mídelo tú mismo,
no lo asumas por la precisión" — es, junto con el reequilibrio dinámico en sí,
la contribución central de este trabajo.

## Arquitectura

- Kernel fusionado (colisión+streaming en una sola pasada, esquema
  "gather-then-relax"), validado antes contra una versión GPU-sola de
  referencia.
- Reparto de dominio por filas: la GPU posee `[0, ny_gpu)`, la CPU posee
  `[ny_gpu, NY)`. Los arrays se reservan siempre a tamaño completo desde el
  arranque para poder redimensionar sin reservar memoria de nuevo.
- Intercambio de halo bidireccional usando datos del paso anterior en ambos
  lados (necesario para permitir solape real de cómputo GPU/CPU sin
  condiciones de carrera).
- Reequilibrio dinámico: cada `REBALANCE_WINDOW` pasos se comparan
  `t_gpu` (medido con `cudaEvent`, tiempo puro de kernel) y `t_cpu` (medido
  con `std::chrono` sobre la región paralelizada con OpenMP), y se recalcula
  el reparto ideal `ny_gpu* = NY · t_cpu / (t_gpu + t_cpu)`, migrando filas
  con `cudaMemcpy2D` solo si el cambio supera una banda muerta
  (`MIN_STEP_ROWS`).
- Capa de auto-calibración (`lbm_heterogeneo_auto.cu`): antes de decidir nada,
  mide por separado el kernel GPU puro (sin ningún concepto de CPU/halo) y el
  kernel heterogéneo (con la rama extra que sabe hablar con la CPU), calcula
  el overhead fijo por paso del esquema heterogéneo, y solo activa el reparto
  si el margen que se ganaría repartiendo supera ese overhead con un factor
  de seguridad. Si no compensa, colapsa a GPU sola y usa a partir de ahí los
  kernels puros — cero coste de CPU/halo residual.

## Resultado principal: FP64

| Versión | NY | Resultado | Referencia GPU sola | Mejora | Validación (diff vs referencia bit-exacta) |
|---|---|---|---|---|---|
| `lbm_heterogeneo_v2.cu` (reparto estático) | 1000 | 627.694 MLUPS | 565.555 MLUPS | +11.0% | 5.718e-15 |
| `lbm_heterogeneo_dinamico.cu` (reequilibrio dinámico) | 1000 | 709.225 MLUPS | 568.795 MLUPS | +24.7% | 9.381e-15 |
| `lbm_heterogeneo_auto.cu` (auto-calibración, versión final corregida) | 1000 | 734.993 MLUPS | ~560 MLUPS | +31.3% | 3.275e-15 |

Los tres son consistentes entre sí (misma arquitectura, refinamientos
sucesivos) y todos están validados frente a la implementación GPU-sola de
referencia con una diferencia del orden de la precisión de máquina en FP64
(`~1e-15`), es decir, no hay divergencia numérica real, solo la esperable por
el orden distinto de las operaciones en coma flotante.

## Investigación FP32: por qué no compensa en este hardware

La hipótesis inicial era que un problema suficientemente grande siempre
acabaría compensando el uso de la CPU, incluso en FP32 (donde la GPU es
~10-13x más rápida que en FP64 pero la CPU solo ~1.3x más rápida). La
investigación mostró que esto es más sutil de lo esperado, en dos capas:

**1. El coste de "saber hablar con la CPU" no es un porcentaje fijo del
kernel.** Se midió por separado el kernel GPU puro (`t_gpu_pura`) y el kernel
heterogéneo con la rama extra de halo (`t_gpu_heterogeneo`), ambos en FP32, y
se repitió la misma configuración (NY=28000) cinco veces seguidas sin pausas:

| Ejecución | Coste extra medido | Margen bruto (`denom`) | Decisión |
|---|---|---|---|
| 1 (arranque en frío) | 2.53% | positivo, pequeño | GPU sola (margen no llega al umbral de seguridad) |
| 2 | 7.44% | negativo | GPU sola (no existe NY que compense) |
| 3 | 8.47% | negativo | GPU sola |
| 4 | 7.98% | negativo | GPU sola |
| 5 | 7.17% | negativo | GPU sola |

Solo la primera ejecución (con la GPU fría) tuvo margen positivo; las cuatro
siguientes, encadenadas sin enfriamiento, midieron un coste 3-4x mayor y
consistentemente negativo. Esto es compatible con un efecto térmico/de reloj
boost de la GPU bajo carga sostenida, agravado por la actividad concurrente
de la CPU (contención de ancho de banda de memoria del sistema durante la
ventana de calibración heterogénea). El "overhead de hablar con la CPU" en
FP32, en este hardware, **no es una constante del kernel: depende del estado
térmico/de reloj en el momento de medir**, y esa variabilidad (2.5%–8.5%) es
del mismo orden que el margen que se intenta explotar.

**2. Ampliar el tamaño del problema no ayuda por sí solo.** El razonamiento
ingenuo ("problema más grande → margen más grande → acaba compensando") no
tiene en cuenta que la calibración reparte un 10% fijo a la CPU, así que un
NY mayor implica más filas absolutas de CPU trabajando en paralelo durante la
calibración — y los datos (NY=20000 → 2000 filas CPU → 2.45% coste extra;
NY=40000 → 4000 filas → 8.89%; NY=100000 → 10000 filas → 12.23%) sugieren que
el coste extra crece con la carga absoluta de CPU, no que se quede fijo. Por
eso NY=100000 no dio mejor resultado que NY=28000: el margen por fila mejora
con NY, pero el coste extra también empeora, y ambos efectos compiten.

**Conclusión honesta:** sobre esta GPU concreta (8GB), no se encontró ningún
NY donde el sistema decida usar CPU+GPU en FP32 de forma robusta y repetible.
Sí se encontró que existe una ventana estrecha (arranque en frío, coste extra
~2.5%) donde el margen es positivo pero insuficiente para superar el umbral
de seguridad exigido. El hallazgo con más valor aquí no es "un NY mágico que
gana", sino que **el sistema nunca elige mal**: en ninguna de las
configuraciones probadas (NY entre 8000 y 100000, cinco repeticiones a
NY=28000) se activó el reparto CPU+GPU cuando no compensaba, es decir, cero
regresiones por decisiones erróneas — el mecanismo de seguridad funciona
exactamente como debía, incluso frente a una fuente de ruido de medida
(variación térmica) que no se había anticipado al diseñarlo.

## Bugs encontrados y corregidos durante el desarrollo

| # | Bug | Síntoma | Cómo se detectó | Fix |
|---|---|---|---|---|
| 1 | Condición de carrera en el halo | Ninguno visible sin validación bit-exacta | Revisión de código + diff contra referencia | Reordenar el snapshot de halo antes de la copia async que lo sobrescribía |
| 2 | Reparto fijo mal ajustado para FP32 | Heterogéneo más lento que GPU sola (2936 vs 7333 MLUPS) | Comparación directa con GPU sola | Reequilibrio dinámico en vez de reparto fijo |
| 3 | Comparar el kernel heterogéneo con el puro como si fueran iguales | Regresión real del 17% a NY=20000 pasando desapercibida como "mejora" | El usuario detectó que el resultado no batía a la GPU sola | Medir `t_gpu_pura` por separado del `t_gpu_heterogeneo` |
| 4 | Factor de seguridad multiplicativo matemáticamente roto | Con margen fino, la condición de activación se volvía imposible de cumplir para cualquier NY | Derivación analítica de la fórmula al buscar un caso FP32 positivo | Cambiar a comparación aditiva `NY·margen > factor·overhead` |
| 5 | Sin comprobación de errores CUDA | A NY=100000 con memoria de GPU insuficiente, el programa "funcionaba" y daba 85936 MLUPS (~10x inflado) sin ningún aviso | El número no encajaba con la banda de rendimiento (8200-9000 MLUPS) vista en todos los demás tamaños | Macro `CUDA_CHECK` en reservas de memoria y puntos críticos, más aviso de memoria libre/necesaria al arrancar |
| 6 | Coste extra del kernel heterogéneo variable con el estado térmico de la GPU | Decisiones distintas para el mismo NY en ejecuciones consecutivas | Repetir la misma configuración 5 veces seguidas | Documentado como limitación/hallazgo, no "arreglado" (es una propiedad real del hardware) |

## Comparación con trabajo previo

La búsqueda de trabajo relacionado (OpenLB, Calore et al. 2017, WaLBerla)
encontró balanceo de carga CPU+GPU para LBM únicamente **estático/offline**:
algoritmos genéticos o auto-tuning de mini-benchmarks ejecutados una vez antes
de la simulación, nunca reequilibrio continuo durante la ejecución. El
reequilibrio dinámico en ventanas de `REBALANCE_WINDOW` pasos, más la capa de
auto-calibración que decide sin intervención humana si conviene repartir en
absoluto (en vez de asumirlo por la precisión usada), no se encontró
documentado en la literatura revisada.

## Limitaciones y trabajo futuro

- Los resultados FP32 son específicos de esta GPU (8GB) y de su
  comportamiento térmico bajo carga sostenida; en otro hardware (GPU con más
  VRAM, con mejor gestión térmica, o con una CPU relativamente más rápida) el
  margen podría ser positivo y estable.
- El coste estructural del kernel heterogéneo (la rama extra que comprueba si
  una dirección viene del halo de CPU) podría reducirse compilando variantes
  especializadas por plantillas en vez de una rama en tiempo de ejecución —
  vía posible para intentar de nuevo un caso FP32 positivo, no explorada en
  esta fase.
- No se ha caracterizado formalmente la relación entre carga de CPU y reloj
  de GPU (la hipótesis de contención térmica/de memoria es razonable dados los
  datos, pero no se ha medido con herramientas de perfilado como `nsys` o
  `nvidia-smi dmon` en paralelo a la ejecución).