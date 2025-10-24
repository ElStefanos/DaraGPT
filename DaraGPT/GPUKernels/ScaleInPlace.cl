__kernel void ScaleInPlace(
    __global float* A,
    const float s,
    const int rows,
    const int cols)
{
    int r = get_global_id(0);
    int c = get_global_id(1);
    if (r >= rows || c >= cols) return;

    int idx = r * cols + c;
    A[idx] = A[idx] * s;
}
