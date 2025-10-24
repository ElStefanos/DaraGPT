__kernel void Transpose2D(
    __global const float* A, // [rows×cols]
    __global float* T,       // [cols×rows]
    const int rows,
    const int cols)
{
    int r = get_global_id(0);
    int c = get_global_id(1);
    if (r >= rows || c >= cols) return;

    T[c * rows + r] = A[r * cols + c];
}
