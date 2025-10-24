__kernel void LayerNormForward(
    __global const float* X,   // [rows*cols]
    __global float* Y,         // [rows*cols]
    __global float* MU,        // [rows] mean
    __global float* INVSTD,    // [rows] 1/sqrt(var+eps)
    int rows, int cols, float eps)
{
    int i = get_global_id(0); // row
    if (i >= rows) return;

    int base = i * cols;

    float mean = 0.0f;
    for (int j = 0; j < cols; ++j) mean += X[base + j];
    mean /= (float)cols;

    float var = 0.0f;
    for (int j = 0; j < cols; ++j) {
        float z = X[base + j] - mean;
        var += z * z;
    }
    var /= (float)cols;

    float inv = 1.0f / sqrt(var + eps);
    MU[i] = mean;
    INVSTD[i] = inv;

    for (int j = 0; j < cols; ++j)
        Y[base + j] = (X[base + j] - mean) * inv;
}

__kernel void LayerNormBackward(
    __global const float* X,       // [rows*cols]
    __global const float* dY,      // [rows*cols]
    __global const float* MU,      // [rows]
    __global const float* INVSTD,  // [rows]
    __global float* dX,            // [rows*cols]
    int rows, int cols)
{
    int i = get_global_id(0); // row
    if (i >= rows) return;

    int base = i * cols;
    float mean = MU[i];
    float inv  = INVSTD[i];

    float sum_dy = 0.0f;
    float sum_dy_xhat = 0.0f;
    for (int j = 0; j < cols; ++j) {
        float xhat = (X[base + j] - mean) * inv;
        float dy   = dY[base + j];
        sum_dy      += dy;
        sum_dy_xhat += dy * xhat;
    }

    for (int j = 0; j < cols; ++j) {
        float xhat = (X[base + j] - mean) * inv;
        float dy   = dY[base + j];
        // dX = (1/N)*inv*(N*dy - sum(dy) - xhat*sum(dy*xhat))
        dX[base + j] = (1.0f/(float)cols) * inv * ((float)cols*dy - sum_dy - xhat*sum_dy_xhat);
    }
}
