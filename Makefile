# ==============================================================================
# Makefile genérico y dinámico para C++ (OpenMP) y CUDA
# Detecta automáticamente archivos .cc y .cu recursivamente dentro de src/
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
TARGET    := $(BIN_DIR)/lbm

# Detectar automáticamente si existe la carpeta 'include' para añadirla
ifneq ($(wildcard include),)
    CXXFLAGS  += -Iinclude
    NVCCFLAGS += -Iinclude
endif

# Función para buscar archivos de manera recursiva en subcarpetas
rwildcard = $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2)$(filter $(subst *,%,$2),$d))

# Buscar todas las fuentes .cc y .cu en 'src/' (ej: src/Phase1_Sequential/main.cc)
CXX_SOURCES := $(call rwildcard,$(SRC_DIR)/,*.cc)
CU_SOURCES  := $(call rwildcard,$(SRC_DIR)/,*.cu)

# Mapear los archivos de código a sus respectivos archivos de objeto en 'build/'
CXX_OBJECTS := $(patsubst $(SRC_DIR)/%.cc,$(BUILD_DIR)/%.o,$(CXX_SOURCES))
CU_OBJECTS  := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SOURCES))

# Detectar si el compilador de NVIDIA (nvcc) está en el sistema
HAS_CUDA := $(shell command -v $(NVCC) 2>/dev/null)

# El enlazador principal siempre será g++ (CXX) para evitar problemas de OpenMP con nvcc
LINKER := $(CXX)

ifeq ($(HAS_CUDA),)
    # Si NO hay CUDA: solo compilamos los fuentes .cc
    OBJECTS := $(CXX_OBJECTS)
else
    # Si SÍ hay CUDA: activamos flag USE_CUDA, añadimos .cu y linkeamos el runtime de CUDA
    CXXFLAGS += -DUSE_CUDA
    OBJECTS  := $(CXX_OBJECTS) $(CU_OBJECTS)
    LDFLAGS  += -L/usr/local/cuda/lib64 -lcudart
endif

# --- REGLAS DE COMPILACIÓN ---

.PHONY: all run clean info

all: $(TARGET)

# Fase de Enlace (Linker)
$(TARGET): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(LINKER) $^ -o $@ $(LDFLAGS)

# Compilar C++ (.cc) recreando la estructura de carpetas en build/
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compilar CUDA (.cu) recreando la estructura de carpetas en build/
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Ejecutar la simulación directamente
run: all
	./$(TARGET)

# Limpiar archivos de compilación y binarios
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Utilidad para comprobar qué está detectando el Makefile
info:
	@echo "========================================================="
	@echo "                INFORMACIÓN DEL PROYECTO"
	@echo "========================================================="
	@echo "CUDA detectado:    $(if $(HAS_CUDA),SÍ (Compilando CPU + GPU),NO (Solo CPU))"
	@echo "Fuentes C++ (.cc): $(CXX_SOURCES)"
	@echo "Fuentes CUDA (.cu):$(CU_SOURCES)"
	@echo "Objetos a crear:   $(OBJECTS)"
	@echo "Destino binario:   $(TARGET)"
	@echo "========================================================="