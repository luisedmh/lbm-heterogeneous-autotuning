#!/usr/bin/env python3
"""
Compara el estado final (array f) del código original y de la versión
fusionada para validar que la fusión de kernels no ha roto la física.

Uso (después de compilar y ejecutar ambos binarios en la misma carpeta,
para que generen f_final_original.bin y f_final.bin):

    python3 compara_resultados.py

Si compilaste lbm_gpu_fused.cu SIN -DUSE_SINGLE_PRECISION (double, la
versión de validación), se espera una diferencia máxima ~1e-12 o menor en
casi todo el dominio, con el error concentrado en la columna de salida
(x = NX-1) y su vecindad inmediata, por el desfase de 1 paso documentado
en el propio .cu. Si el error aparece disperso por TODO el dominio y no
solo cerca de x=NX-1, hay un bug real que hay que investigar antes de
fiarte de los tiempos.

Si compilaste con -DUSE_SINGLE_PRECISION, cambia DTYPE_FUSED a 'float32'
más abajo: ahí ya no tiene sentido esperar coincidencia bit a bit (es otra
precisión), solo que el campo de velocidades sea físicamente razonable
(compara magnitudes, no bits).
"""
import numpy as np
import sys

NX, NY = 1000, 1000
Q = 9

DTYPE_ORIGINAL = 'float64'
DTYPE_FUSED = 'float64'   # cambia a 'float32' si compilaste con -DUSE_SINGLE_PRECISION

def load(path, dtype):
    arr = np.fromfile(path, dtype=dtype)
    expected = Q * NX * NY
    if arr.size != expected:
        print(f"AVISO: {path} tiene {arr.size} elementos, se esperaban {expected}")
        sys.exit(1)
    return arr.reshape(Q, NY, NX)

try:
    f_orig = load('f_final_original.bin', DTYPE_ORIGINAL)
    f_new = load('f_final.bin', DTYPE_FUSED)
except FileNotFoundError as e:
    print(f"No encuentro el fichero: {e}. Ejecuta primero los dos binarios en esta carpeta.")
    sys.exit(1)

if DTYPE_ORIGINAL != DTYPE_FUSED:
    f_orig = f_orig.astype(np.float64)
    f_new = f_new.astype(np.float64)

diff = np.abs(f_orig - f_new)

print(f"Diferencia maxima absoluta global : {diff.max():.3e}")
print(f"Diferencia media absoluta global  : {diff.mean():.3e}")

# Diferencia máxima por columna x, para confirmar que se concentra en x=NX-1
diff_por_columna = diff.max(axis=(0, 1))  # shape (NX,)
peores_columnas = np.argsort(diff_por_columna)[::-1][:10]
print("\nColumnas x con mayor diferencia (se espera que sea NX-1=999 y alrededores):")
for x in peores_columnas:
    print(f"  x={x:4d}  diff_max={diff_por_columna[x]:.3e}")

# Diferencia excluyendo la última columna (la del desfase documentado)
diff_sin_salida = diff[:, :, :NX - 1]
print(f"\nDiferencia maxima excluyendo la columna de salida (x<{NX-1}): {diff_sin_salida.max():.3e}")
print("Si este ultimo numero es ~1e-12 o menor (double), la fusion es correcta")
print("y toda la diferencia esta, como se esperaba, aislada en la frontera de salida.")
