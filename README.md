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
