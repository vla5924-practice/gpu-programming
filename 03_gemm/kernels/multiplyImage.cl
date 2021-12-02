#define BLOCK_SIZE 16

/**
 * Kernel multiplyImage works for square matrices only
 */

__kernel void multiplyImage(__read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t c, int m, int n, int k) {
    __local float A[BLOCK_SIZE][BLOCK_SIZE];
    __local float B[BLOCK_SIZE][BLOCK_SIZE];
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);
    int row = get_global_id(0);
    int col = get_global_id(1);
    int blocks = m / BLOCK_SIZE;
    float s = 0;
    for (int i = 0; i < blocks; i++) {
        float x = read_imagef(a, (int2)(BLOCK_SIZE * i + local_row, col)).x;
        float y = read_imagef(b, (int2)(row, BLOCK_SIZE * i + local_col)).x;
        A[local_col][local_row] = x;
        B[local_col][local_row] = y;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int j = 0; j < BLOCK_SIZE; j++)
            s += A[local_col][j] * B[j][local_row];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    write_imagef(c, (int2)(row, col), s);
}
