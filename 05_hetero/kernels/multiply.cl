#define BLOCK_SIZE 16

__kernel void multiply(__global float *a, __global float *b, __global float *c, int n) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    __local float A[BLOCK_SIZE][BLOCK_SIZE];
    __local float B[BLOCK_SIZE][BLOCK_SIZE];
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);
    int blocks = n / BLOCK_SIZE;
    float s = 0;
    for (int i = 0; i < blocks; i++) {
        A[local_col][local_row] = a[col * n + BLOCK_SIZE * i + local_row];
        B[local_col][local_row] = b[(BLOCK_SIZE * i + local_col) * n + row];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = 0; j < BLOCK_SIZE; j++)
            s += A[local_col][j] * B[j][local_row];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[col * n + row] = s;
}
