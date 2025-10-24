__kernel void RowSoftmax(
    __global const float* A,  // [rows×cols]
    __global float* Out,      // [rows×cols]
    const int rows,
    const int cols)
{
    int r = get_global_id(0);
    if (r >= rows) return;

    // find max
    float mx = -MAXFLOAT;
    int base = r * cols;
    for (int c = 0; c < cols; c++)
        mx = fmax(mx, A[base + c]);

    // sum exp
    float sum = 0.0f;
    for (int c = 0; c < cols; c++)
        sum += exp(A[base + c] - mx);

    float inv = 1.0f / sum;
    for (int c = 0; c < cols; c++)
        Out[base + c] = exp(A[base + c] - mx) * inv;
}

__kernel void RowSoftmaxBackward(
    __global const float* A,      
    __global const float* dA,     
    __global float* dScores,     
    int rows, int cols)
{
    int i = get_global_id(0); // row
    if (i >= rows) return;

    // sum_j dA[i,j] * A[i,j]
    float dot = 0.0f;
    int base = i * cols;
    for (int j = 0; j < cols; ++j)
        dot += dA[base + j] * A[base + j];

    for (int j = 0; j < cols; ++j) {
        float y = A[base + j];
        dScores[base + j] = (dA[base + j] - dot) * y;
    }
}
