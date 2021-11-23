#define BLOCK_SIZE 16

__kernel void multiplyImage(__read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t c, int m, int n, int k) {
    __local float A[BLOCK_SIZE][BLOCK_SIZE];
    __local float B[BLOCK_SIZE][BLOCK_SIZE];
    int local_row = get_local_id(1);
    int local_col = get_local_id(0);
    int row = get_global_id(0);
    int col = get_global_id(1);
    int blocks = m / BLOCK_SIZE;
    float s = 0;
    for (int i = 0; i < blocks; i++) {
        int2 idxA = {BLOCK_SIZE * i + local_col, col};
        int2 idxB = {row, BLOCK_SIZE * i + local_row};
        A[local_row][local_col] = read_imagef(a, idxA).x;
        B[local_row][local_col] = read_imagef(b, idxB).x;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = 0; j < BLOCK_SIZE; j++)
            s += A[local_row][j] * B[j][local_col];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int2 idxC = {row, col};
    write_imagef(c, idxC, s);
}

__kernel void multiplyImageNaive(__read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t c, int m, int n, int k) {
    __local float A[BLOCK_SIZE][BLOCK_SIZE];
    __local float B[BLOCK_SIZE][BLOCK_SIZE];
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);
    int row = get_global_id(0);
    int col = get_global_id(1);
    int blocks = m / BLOCK_SIZE;
    float s = 0;
    for (int i = 0; i < blocks; i++) {
        int2 idxA = {row, BLOCK_SIZE * i + local_col};
        int2 idxB = {BLOCK_SIZE * i + local_row, col};
        A[local_row][local_col] = read_imagef(a, idxA).x;
        B[local_row][local_col] = read_imagef(b, idxB).x;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = 0; j < BLOCK_SIZE; j++)
            s += A[local_row][j] * B[j][local_col];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int2 idxC = {row, col};
    write_imagef(c, idxC, (float4)(s, 0, 0, 0));
}
