__kernel void MatMulRM(
    __global const float* A,   // [m×k]
    __global const float* B,   // [k×n]
    __global float* C,         // [m×n]
    const int m,
    const int k,
    const int n)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= m || col >= n) return;

    float sum = 0.0f;
    int aBase = row * k;
    for (int t = 0; t < k; t++)
        sum += A[aBase + t] * B[t * n + col];

    C[row * n + col] = sum;
}
