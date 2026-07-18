# ==============================================================================
# Makefile genérico y dinámico para C++ (OpenMP) y CUDA
# Organizado por fases: cada subcarpeta directa de src/ es una fase independiente
# con su propio main() (ej. src/Phase1_Sequential, src/Phase2_OpenMP, ...).
#
# Uso:
#   make build            -> menú interactivo para elegir fase y compilarla
#   make run              -> menú interactivo para elegir fase, compilarla y ejecutarla
#   make clean            -> menú interactivo para elegir fase a limpiar (o "todas")
#   make build PHASE=1                    -> compila la fase nº1 del menú, sin preguntar
#   make build PHASE=Phase1_Sequential    -> compila esa fase por nombre, sin preguntar
#   make clean PHASE=all                  -> limpia build/ y bin/ por completo
#   make info              -> muestra las fases detectadas (y detalles si pasas PHASE=)
# ==============================================================================

CXX       := g++
NVCC      := nvcc
CXXFLAGS  := -O3 -Wall -Wextra -fopenmp
NVCCFLAGS := -O3 -Xcompiler -fopenmp
LDFLAGS   := -fopenmp -lm

# Directorios del proyecto
SRC_DIR   := src
BUILD_DIR := build
BIN_DIR   := bin

# Detectar automáticamente si existe la carpeta 'include' para añadirla
ifneq ($(wildcard include),)
    CXXFLAGS  += -Iinclude
    NVCCFLAGS += -Iinclude
endif

# Función para buscar archivos de manera recursiva en subcarpetas
rwildcard = $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2)$(filter $(subst *,%,$2),$d))

# Detectar si el compilador de NVIDIA (nvcc) está en el sistema
HAS_CUDA := $(shell command -v $(NVCC) 2>/dev/null)

# El enlazador principal siempre será g++ (CXX) para evitar problemas de OpenMP con nvcc
LINKER := $(CXX)

# --- FASES ---
# Cada subcarpeta directa de src/ es una fase (ej. Phase1_Sequential, Phase2_OpenMP)
PHASE_DIRS  := $(sort $(notdir $(patsubst %/,%,$(wildcard $(SRC_DIR)/*/))))
NUM_PHASES  := $(words $(PHASE_DIRS))

# Resuelve la fase a usar en build/run: usa PHASE= si viene dada (nombre o número),
# si no, muestra un menú interactivo. Imprime el nombre de la fase resuelta a stdout.
define RESOLVE_PHASE
if [ -n "$(PHASE)" ]; then \
	req="$(PHASE)"; \
else \
	i=1; \
	for p in $(PHASE_DIRS); do echo "  $$i) $$p" >&2; i=$$((i+1)); done; \
	printf "Selecciona una fase [1-$(NUM_PHASES)]: " >&2; \
	read req; \
fi; \
if echo "$$req" | grep -qE '^[0-9]+$$'; then \
	sel=$$(echo "$(PHASE_DIRS)" | tr ' ' '\n' | sed -n "$${req}p"); \
else \
	sel=""; \
	for p in $(PHASE_DIRS); do [ "$$p" = "$$req" ] && sel="$$p"; done; \
fi; \
if [ -z "$$sel" ]; then \
	echo "Fase inválida: '$$req'. Fases disponibles: $(PHASE_DIRS)" >&2; \
else \
	echo "$$sel"; \
fi
endef

# Igual que RESOLVE_PHASE pero además acepta "ALL"/"all" (por número o por nombre),
# usado por 'make clean' para poder limpiar todas las fases de golpe.
define RESOLVE_PHASE_ALL
if [ -n "$(PHASE)" ]; then \
	req="$(PHASE)"; \
else \
	i=1; \
	for p in $(PHASE_DIRS); do echo "  $$i) $$p" >&2; i=$$((i+1)); done; \
	echo "  $$i) Todas las fases" >&2; \
	printf "Selecciona una fase [1-$$i]: " >&2; \
	read req; \
fi; \
reqlc=$$(echo "$$req" | tr '[:upper:]' '[:lower:]'); \
if [ "$$reqlc" = "all" ]; then \
	echo "ALL"; \
elif echo "$$req" | grep -qE '^[0-9]+$$'; then \
	if [ "$$req" -eq $$(( $(NUM_PHASES) + 1 )) ]; then \
		echo "ALL"; \
	else \
		sel=$$(echo "$(PHASE_DIRS)" | tr ' ' '\n' | sed -n "$${req}p"); \
		if [ -z "$$sel" ]; then \
			echo "Fase inválida: '$$req'." >&2; \
		else \
			echo "$$sel"; \
		fi; \
	fi; \
else \
	sel=""; \
	for p in $(PHASE_DIRS); do [ "$$p" = "$$req" ] && sel="$$p"; done; \
	if [ -z "$$sel" ]; then \
		echo "Fase inválida: '$$req'. Fases disponibles: $(PHASE_DIRS) all" >&2; \
	else \
		echo "$$sel"; \
	fi; \
fi
endef

# --- Fuentes/objetos/target de la fase actual (solo válido cuando PHASE está fijado) ---
CXX_SOURCES := $(if $(PHASE),$(call rwildcard,$(SRC_DIR)/$(PHASE)/,*.cc))
CU_SOURCES  := $(if $(PHASE),$(call rwildcard,$(SRC_DIR)/$(PHASE)/,*.cu))

CXX_OBJECTS := $(patsubst $(SRC_DIR)/%.cc,$(BUILD_DIR)/%.o,$(CXX_SOURCES))
CU_OBJECTS  := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SOURCES))

# Detectar .cc que en realidad contienen código CUDA (kernels __global__, __constant__,
# o que incluyen cuda_runtime.h). Estos archivos se compilarán con nvcc como si fueran .cu.
CXX_CUDA_SOURCES := $(if $(CXX_SOURCES),$(shell grep -lE '__global__|__device__|__constant__|cuda_runtime\.h' $(CXX_SOURCES) 2>/dev/null))

ifneq ($(and $(HAS_CUDA),$(or $(CU_SOURCES),$(CXX_CUDA_SOURCES))),)
    # Si hay CUDA y la fase tiene fuentes .cu (o .cc con código CUDA): activamos flag
    # USE_CUDA y linkeamos el runtime
    CXXFLAGS += -DUSE_CUDA
    OBJECTS  := $(CXX_OBJECTS) $(CU_OBJECTS)
    LDFLAGS  += -L/usr/local/cuda/lib64 -lcudart
else
    OBJECTS := $(CXX_OBJECTS)
endif

TARGET := $(BIN_DIR)/lbm_$(PHASE)

# --- REGLAS DE COMPILACIÓN ---

.PHONY: all build run clean info check-phase _build _run _clean

all: build

# Targets visibles: resuelven la fase (interactiva o vía PHASE=) y relanzan make
build:
	@phase=$$($(RESOLVE_PHASE)); \
	[ -n "$$phase" ] || exit 1; \
	$(MAKE) --no-print-directory PHASE="$$phase" _build

run:
	@phase=$$($(RESOLVE_PHASE)); \
	[ -n "$$phase" ] || exit 1; \
	$(MAKE) --no-print-directory PHASE="$$phase" _run

clean:
	@phase=$$($(RESOLVE_PHASE_ALL)); \
	[ -n "$$phase" ] || exit 1; \
	if [ "$$phase" = "ALL" ]; then \
		rm -rf $(BUILD_DIR) $(BIN_DIR); \
		echo "Limpiadas todas las fases."; \
	else \
		$(MAKE) --no-print-directory PHASE="$$phase" _clean; \
	fi

# Evita que _build/_run/_clean se invoquen directamente sin haber resuelto una fase
check-phase:
	@if [ -z "$(PHASE)" ]; then \
		echo "Error: usa 'make build', 'make run' o 'make clean' (no invoques este target directamente)."; \
		exit 1; \
	fi

# Fase de Enlace (Linker)
_build: check-phase $(TARGET)

$(TARGET): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(LINKER) $^ -o $@ $(LDFLAGS)

# Compilar C++ (.cc) recreando la estructura de carpetas en build/
# Si el propio .cc contiene código CUDA (kernels __global__, cuda_runtime.h, etc.),
# se compila con nvcc forzando el lenguaje CUDA (-x cu) para que tenga acceso al
# runtime y a la sintaxis de kernels; si no, se compila normalmente con g++.
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cc
	@mkdir -p $(dir $@)
	@if [ -n "$(HAS_CUDA)" ] && grep -qE '__global__|__device__|__constant__|cuda_runtime\.h' $<; then \
		echo "$(NVCC) -x cu ... -c $< -o $@ (detectado código CUDA en .cc)"; \
		$(NVCC) -x cu $(NVCCFLAGS) -DUSE_CUDA -c $< -o $@; \
	else \
		$(CXX) $(CXXFLAGS) -c $< -o $@; \
	fi

# Compilar CUDA (.cu) recreando la estructura de carpetas en build/
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Ejecutar la simulación de la fase seleccionada
_run: _build
	./$(TARGET)

# Limpiar archivos de compilación y binario de la fase seleccionada
_clean: check-phase
	rm -rf $(BUILD_DIR)/$(PHASE) $(BIN_DIR)/lbm_$(PHASE)
	@echo "Limpiada la fase $(PHASE)."

# Utilidad para comprobar qué está detectando el Makefile
info:
	@echo "========================================================="
	@echo "                INFORMACIÓN DEL PROYECTO"
	@echo "========================================================="
	@echo "Fases detectadas: $(PHASE_DIRS)"
	@echo "CUDA detectado:    $(if $(HAS_CUDA),SÍ (nvcc disponible),NO (solo CPU))"
ifneq ($(PHASE),)
	@echo "---------------------------------------------------------"
	@echo "Fase seleccionada (PHASE=$(PHASE)):"
	@echo "  Fuentes C++ (.cc): $(CXX_SOURCES)"
	@echo "  Fuentes CUDA (.cu):$(CU_SOURCES)"
	@echo "  Objetos a crear:   $(OBJECTS)"
	@echo "  Destino binario:   $(TARGET)"
endif
	@echo "========================================================="
