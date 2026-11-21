# Post-Contenido 1: Benchmark CUDA CPU vs GPU y Análisis de Aceleradores de IA

**Asignatura:** Arquitectura de Computadores — Unidad 11  


---
## 1. Descripción del Entorno de Ejecución

Dado que el entorno local se ejecuta sobre VirtualBox sin acceso directo a una GPU física dedicada, las pruebas de compilación y ejecución de kernels CUDA se realizaron en **Google Colab** utilizando una GPU NVIDIA Tesla T4, siguiendo las indicaciones de la guía de laboratorio:

* **GPU:** NVIDIA Tesla T4 (16 GB GDDR6, Arquitectura Turing, Compute Capability 7.5) *(verificar salida de `nvidia-smi`)*
* **Versión de CUDA Toolkit:** CUDA 12.x (`nvcc --version` -> [ej. release 12.2, V12.2.140])
* **Compilador Host:** GCC [ej. 11.4.0] (`gcc --version`)
* **Sistema Operativo:** Ubuntu 22.04 LTS (Entorno Google Colab / VirtualBox para gestión Git)

---

## 2. Estructura del Repositorio

```text
apellido-post1-u11/
├── src/
│   ├── vectorAdd.cu          # Suma de vectores CPU vs GPU (1M, 4M, 16M elementos)
│   ├── matMul.cu             # Multiplicación de matrices Naïve vs Tiling (Shared Memory)
│   └── vectorAddStream.cu    # Transferencia síncrona vs asíncrona con CUDA Streams
├── capturas/                 # Evidencias de ejecución de los checkpoints
├── docs/
│   └── analisis-comparativo.md  # Análisis técnico de aceleradores de IA (Parte 2)
└── README.md                 # Documentación principal y resultados de la Parte 1
```

---

## 3. Comandos de Compilación y Ejecución

Todos los programas fueron compilados con `nvcc` aplicando optimización de nivel `-O2`:

```bash
# 1. Suma de vectores (Checkpoint 1)
nvcc -O2 -o vectorAdd src/vectorAdd.cu
./vectorAdd

# 2. Multiplicación de matrices con Shared Memory (Checkpoint 2)
nvcc -O2 -o matMul src/matMul.cu
./matMul

# 3. Suma de vectores con CUDA Streams y Pinned Memory (Checkpoint 4)
nvcc -O2 -o vectorAddStream src/vectorAddStream.cu
./vectorAddStream
```

---

## 4. Resultados de Benchmarks (Parte 1)

### 4.1. Suma de Vectores (`vectorAdd.cu`)

Comparativa de tiempos de ejecución entre CPU secuencial, el tiempo aislado del kernel en GPU (`cudaEvent`) y el tiempo total en GPU incluyendo las transferencias de memoria `cudaMemcpy` (Host-to-Device y Device-to-Host):

| Tamaño del Vector ($N$) | Memoria Total Transferida | Tiempo CPU (ms) | Tiempo GPU Kernel (ms) | Tiempo GPU Total con `cudaMemcpy` (ms) | Speedup (Solo Kernel) | Verificación |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1M** ($2^{20}$) | ~12 MB | `[ej. 2.15]` | `[ej. 0.14]` | `[ej. 2.85]` | `[ej. 15.3x]` | `Errores: 0` |
| **4M** ($2^{22}$) | ~48 MB | `[ej. 8.90]` | `[ej. 0.48]` | `[ej. 8.20]` | `[ej. 18.5x]` | `Errores: 0` |
| **16M** ($2^{24}$) | ~192 MB | `[ej. 35.40]` | `[ej. 1.82]` | `[ej. 21.50]` | `[ej. 19.4x]` | `Errores: 0` |

### 4.2. Multiplicación de Matrices (`matMul.cu` — Naïve vs. Tiling `TILE=16`)

Comparativa de rendimiento para matrices cuadradas de tamaño $N \times N$, evaluando el impacto de reducir los accesos redundantes a memoria global mediante memoria compartida (`__shared__`) con bloques de $16 \times 16$:

| Dimensión ($N \times N$) | Tiempo CPU (ms) | GPU Naïve Sin Tiling (ms) | GPU Tiling `TILE=16` (ms) | Speedup (Naïve vs. Tiling) | Error Máximo |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **512 × 512** | `[ej. 410.20]` | `[ej. 2.45]` | `[ej. 0.85]` | `[ej. 2.88x]` | `< 1e-3` |
| **1024 × 1024** | `[ej. 3850.60]` | `[ej. 18.90]` | `[ej. 5.60]` | `[ej. 3.37x]` | `< 1e-3` |

### 4.3. Transferencia Síncrona vs. Asíncrona con Streams (`vectorAddStream.cu`)

Evaluación del solapamiento entre transferencia de datos por PCIe y ejecución de cómputo utilizando memoria page-locked (`cudaHostAlloc`) y `NSTREAMS = 4` para $N = 16\text{M}$ elementos:

| Estrategia de Ejecución | Tiempo Total Medido (ms) | Reducción de Tiempo (%) |
| :--- | :---: | :---: |
| **Síncrona** (`cudaMemcpy` bloqueante + kernel) | `[ej. 14.50]` | Línea base (0%) |
| **Asíncrona con Streams** (`NSTREAMS = 4`, solapado) | `[ej. 12.80]` | `[ej. 11.7%]` |

---

## 5. Análisis de Resultados

Para volúmenes de datos grandes (como $N = 16\text{M}$ elementos en `vectorAdd` o matrices de $1024 \times 1024$ en `matMul`), el kernel de la GPU supera ampliamente a la CPU debido a su arquitectura masivamente paralela (SIMT). Mientras que una CPU ejecuta la operación de forma secuencial o con un número reducido de hilos vectoriales, la GPU despliega miles de hilos concurrentes organizados en warps y bloques que ocultan la latencia de acceso a memoria mediante cambio rápido de contexto en hardware. Adicionalmente, en `matMulTiled`, el uso de memoria compartida (`__shared__`) permite reutilizar cada submatriz cargada 16 veces dentro de los Streaming Multiprocessors (SM), evitando el cuello de botella de leer repetidamente desde la DRAM global de la tarjeta.

Sin embargo, cuando se analiza el tiempo total de ejecución de la GPU incluyendo las llamadas a `cudaMemcpy`, se observa que para tamaños pequeños ($N = 1\text{M}$ o menores) la GPU puede resultar incluso más lenta que la CPU. Esto ocurre porque el bus PCI Express introduce una latencia fija de inicialización y un límite de ancho de banda al mover los arreglos desde la memoria principal del host (RAM) hacia la memoria del dispositivo (VRAM) y viceversa. En operaciones de baja complejidad aritmética como `vectorAdd`, el tiempo invertido en transportar los datos por PCIe supera por un orden de magnitud al tiempo real de cálculo en los núcleos CUDA, lo que demuestra que acelerar un algoritmo en GPU solo es rentable cuando la carga computacional compensa el costo de transferencia de memoria.

---

## 6. Decisiones Técnicas Justificadas (Parte 1)

### 6.1. ¿Conviene `cudaMemcpyAsync` con streams o transferencia síncrona para `vectorAdd`?

La operación `vectorAdd` presenta una intensidad aritmética extremadamente baja (1 operación de punto flotante por cada 12 bytes transferidos: dos lecturas de 4 bytes y una escritura de 4 bytes), lo que la clasifica como un problema estrictamente limitado por memoria (*memory-bound*). Para $N = 16\text{M}$ elementos, el volumen total transferido entre host y dispositivo asciende a aproximadamente 192 MB (64 MB por cada arreglo de entrada `A` y `B`, más 64 MB para el resultado `C`). Sobre un bus PCIe con un ancho de banda efectivo típico de 10 a 12 GB/s, mover esos 192 MB requiere entre 16 ms y 19 ms, un tiempo un orden de magnitud superior a los ~1.8 ms que tarda en ejecutarse el kernel de suma. Dado que el uso de CUDA streams no elimina la transferencia de datos sino que únicamente permite solapar la copia de un fragmento con el cómputo de otro, el ahorro máximo teórico está acotado por la fase más corta del pipeline, es decir, los ~1.8 ms de cómputo. En consecuencia, incluso un solapamiento ideal solo logra ocultar alrededor de un 10% del tiempo total, por lo que implementar streams y memoria *pinned* en `vectorAdd` ofrece un beneficio marginal. En contraste, esta técnica resulta altamente conveniente en operaciones intensivas en cómputo (*compute-bound*) como `matMulTiled`, donde la complejidad $O(N^3)$ hace que el tiempo de ejecución del kernel sea comparable o superior al de transferencia, permitiendo ocultar casi por completo el costo del bus PCIe.

### 6.2. ¿Cuántos streams usar y qué limita la ganancia al aumentar su número?

Incrementar el parámetro `NSTREAMS` de 4 a 16 manteniendo fijo el tamaño del problema ($N = 16\text{M}$) no produce una reducción proporcional en el tiempo total de ejecución, sino que alcanza rápidamente un punto de rendimientos decrecientes. Esta limitación se debe a que los streams son colas lógicas de ejecución en software que compiten por recursos físicos finitos y compartidos en el hardware de la GPU. En primer lugar, la mayoría de las GPUs NVIDIA modernas disponen únicamente de 1 o 2 motores de copia DMA (*copy engines*) encargados de gestionar las transferencias PCIe de forma independiente a los motores de cómputo (permitiendo como máximo una transferencia Host-to-Device y una Device-to-Host simultáneas). Por ello, utilizar entre 2 y 4 streams suele ser suficiente para mantener saturado el pipeline de transferencia y cómputo; añadir más streams no crea motores DMA adicionales y, por el contrario, introduce sobrecarga de planificación y lanzamiento por cada fragmento extra. En segundo lugar, el ancho de banda del bus PCIe es un recurso físico único compartido por todos los streams activos: subdividir los 192 MB en 16 bloques más pequeños no reduce el tiempo agregado de transmisión de los datos, sino que únicamente reorganiza su distribución temporal.

---
