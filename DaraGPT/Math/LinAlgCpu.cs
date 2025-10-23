namespace DaraGPT;

public static class LinAlgCPU
{
    public static float[] MatMul(float[] A, float[] B, int m, int n, int p)
    {
        var outArr = new float[m * p];
        for (var i = 0; i < m; i++)
        for (var k = 0; k < n; k++)
        {
            var av = A[i * n + k];
            for (var j = 0; j < p; j++) outArr[i * p + j] += av * B[k * p + j];
        }

        return outArr;
    }

    public static float[] SoftmaxRowwise(float[] X, int rows, int cols)
    {
        var outA = new float[X.Length];
        for (var r = 0; r < rows; r++)
        {
            var max = -float.MaxValue;
            var baseIdx = r * cols;
            for (var c = 0; c < cols; c++) max = Math.Max(max, X[baseIdx + c]);
            double sum = 0;
            for (var c = 0; c < cols; c++)
            {
                var v = Math.Exp(X[baseIdx + c] - max);
                outA[baseIdx + c] = (float)v;
                sum += v;
            }

            for (var c = 0; c < cols; c++) outA[baseIdx + c] /= (float)sum;
        }

        return outA;
    }

    public static float[] MatMulTranspose(float[] A, float[] B, int m, int n, int p)
    {
        // A: (n x m) implicitno (transponovana od (m x n))
        // B: (n x p)
        // rezultat: (m x p)
        var result = new float[m * p];
        for (var i = 0; i < m; i++)
        for (var j = 0; j < p; j++)
        {
            float sum = 0;
            for (var k = 0; k < n; k++)
                sum += A[k * m + i] * B[k * p + j];
            result[i * p + j] = sum;
        }

        return result;
    }


    public static void Transpose(float[] x, int rows, int cols, float[] y)
    {
        for (var r = 0; r < rows; r++)
        for (var c = 0; c < cols; c++)
            y[c * rows + r] = x[r * cols + c];
    }
}