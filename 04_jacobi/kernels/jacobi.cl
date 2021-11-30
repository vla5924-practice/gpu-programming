__kernel void jacobi(__global float *a, __global float *b, __global float *x0, __global float *x1, int n) {
    int i = get_global_id(0);
    float s = 0;
    for (int j = 0; j < n; j++)
        s += i != j ? a[j * n + i] * x0[j] : 0;
    x1[i] = (b[i] - s) / a[i * n + i];
}
