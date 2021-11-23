#define BLOCK_SIZE 16

__kernel void multiplyBlock(__global float *a, __global float *b, __global float *c, int m, int n, int k) {
    __local float A[BLOCK_SIZE][BLOCK_SIZE];
    __local float B[BLOCK_SIZE][BLOCK_SIZE];
    int local_row = get_local_id(1);
    int local_col = get_local_id(0);
    int row = get_global_id(0);
    int col = get_global_id(1);
    int blocks = m / BLOCK_SIZE;
    float s = 0;
    for (int i = 0; i < blocks; i++) {
        A[local_row][local_col] = a[col * m + BLOCK_SIZE * i + local_col];
        B[local_row][local_col] = b[(BLOCK_SIZE * i + local_row) * k + row];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = 0; j < BLOCK_SIZE; j++)
            s += A[local_row][j] * B[j][local_col];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[col * k + row] = s;
}

__kernel void multiplyBlockNaive(__global float *a, __global float *b, __global float *c, int m, int n, int k) {
    __local float A[BLOCK_SIZE][BLOCK_SIZE];
    __local float B[BLOCK_SIZE][BLOCK_SIZE];
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);
    int row = get_global_id(0);
    int col = get_global_id(1);
    int blocks = m / BLOCK_SIZE;
    float s = 0;
    for (int i = 0; i < blocks; i++) {
        A[local_row][local_col] = a[row * n + BLOCK_SIZE * i + local_col];
        B[local_row][local_col] = b[(BLOCK_SIZE * i + local_row) * k + col];
        barrier(CLK_GLOBAL_MEM_FENCE);
        for (int j = 0; j < BLOCK_SIZE; j++)
            s += A[local_row][j] * B[j][local_col];
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
    c[row * k + col] = s;
}
