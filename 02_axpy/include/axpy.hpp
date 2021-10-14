#include <CL/cl.h>

void saxpy(int n, float a, float *x, int incx, float *y, int incy);
void daxpy(int n, double a, double *x, int incx, double *y, int incy);

void saxpy_ocl(int n, float a, float *x, int incx, float *y, int incy, cl_device_id deviceId, double *elapsed = nullptr);
void daxpy_ocl(int n, double a, double *x, int incx, double *y, int incy, cl_device_id deviceId, double *elapsed = nullptr);

void saxpy_omp(int n, float a, float *x, int incx, float *y, int incy);
void daxpy_omp(int n, double a, double *x, int incx, double *y, int incy);
