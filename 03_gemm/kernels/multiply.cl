__kernel void multiply(__global float *a, __global float *b, __global float *c, int m, int n, int k) {
    int row = get_global_id(1);
    int col = get_global_id(0);
    float s = 0;
    int i;
    for (i = 0; i < n; i++)
        s += a[row * n + i] * b[col + n * i];
    c[k * row + col] = s;
}
