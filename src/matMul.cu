#define TILE 16

__global__ void matMulTiled(const float *A, const float *B,
                             float *C, int N) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; t++) {
        // Cargar tile de A en shared memory
        if (row < N && t*TILE + threadIdx.x < N)
            sA[threadIdx.y][threadIdx.x] =
                A[row*N + t*TILE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        // Cargar tile de B en shared memory
        if (col < N && t*TILE + threadIdx.y < N)
            sB[threadIdx.y][threadIdx.x] =
                B[(t*TILE + threadIdx.y)*N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();   // Esperar a que el tile esté cargado

        for (int k = 0; k < TILE; k++)
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();   // Esperar antes del siguiente tile
    }

    if (row < N && col < N) C[row*N + col] = sum;
    
    // Lanzamiento: grid 2D de tiles
dim3 block(TILE, TILE);
dim3 grid((N+TILE-1)/TILE, (N+TILE-1)/TILE);
matMulTiled<<<grid, block>>>(d_A, d_B, d_C, N);
}


