__kernel void calculateArray(__global int *arr)
{
	int global_id = get_global_id(0);
	arr[global_id] += global_id;
}
