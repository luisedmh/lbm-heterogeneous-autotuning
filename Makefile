# Makefile genérico: no hay que listar archivos a mano.
# Cualquier .c o .cu que exista bajo src/ (en cualquier subcarpeta) se compila solo.

CC        := gcc
NVCC      := nvcc
CFLAGS    := -O3 -Wall -Wextra -fopenmp -Iinclude
NVCCFLAGS := -O3 -Iinclude -Xcompiler -fopenmp
LDFLAGS   := -fopenmp -lm

SRC_DIR   := src
BUILD_DIR := build
BIN       := bin/lbm

# Wildcard recursivo (GNU Make no tiene ** nativo, esta es la forma estándar de conseguirlo)
rwildcard = $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2)$(filter $(subst *,%,$2),$d))

C_SOURCES  := $(call rwildcard,$(SRC_DIR)/,*.c)
CU_SOURCES := $(call rwildcard,$(SRC_DIR)/,*.cu)

C_OBJECTS  := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(C_SOURCES))
CU_OBJECTS := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SOURCES))

# Detecta si nvcc está disponible. Si no lo está, el proyecto compila igualmente
# y se queda solo con el backend de CPU (así es usable en cualquier máquina).
HAS_CUDA := $(shell command -v $(NVCC) 2>/dev/null)

ifeq ($(HAS_CUDA),)
  OBJECTS := $(C_OBJECTS)
  LINKER  := $(CC)
else
  CFLAGS  += -DUSE_CUDA
  OBJECTS := $(C_OBJECTS) $(CU_OBJECTS)
  LDFLAGS += -lcudart
  LINKER  := $(NVCC)
endif

.PHONY: all run clean info

all: $(BIN)

$(BIN): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(LINKER) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cc
        @mkdir -p $(dir $@)
        $(NVCC) $(NVCCFLAGS) -c $< -o $@

run: all
	./$(BIN) --backend cpu

clean:
	rm -rf $(BUILD_DIR) bin

info:
	@echo "CUDA detectado: $(if $(HAS_CUDA),si ($(HAS_CUDA)),no -- solo se compila el backend CPU)"
	@echo "Fuentes C:  $(C_SOURCES)"
	@echo "Fuentes CU: $(CU_SOURCES)"
