__kernel void saxpy(int n, float a, __global float *x, int incx, __global float *y, int incy) {
    int gid = get_global_id(0);
    y[gid * incy] += a * x[gid * incx];
}

__kernel void daxpy(int n, double a, __global double *x, int incx, __global double *y, int incy) {
    int gid = get_global_id(0);
    y[gid * incy] += a * x[gid * incx];
}
