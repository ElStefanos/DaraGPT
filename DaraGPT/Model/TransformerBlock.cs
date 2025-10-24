using DaraGPT.Math.Gpu;

namespace DaraGPT.Model;

public class TransformerBlock
{
    private GpuEngine _gpu;
    public FeedForward Ffn;
    public MultiHeadAttention Mha;

    public TransformerBlock(int dModel, int numHeads, GpuEngine gpu)
    {
        Mha = new MultiHeadAttention(dModel, numHeads, gpu);
        Ffn = new FeedForward(dModel, dModel * 4);
        _gpu = gpu;
    }

    public float[,] FfnW1
    {
        get => Ffn.W1;
        set => Ffn.W1 = value;
    }

    public float[,] FfnW2
    {
        get => Ffn.W2;
        set => Ffn.W2 = value;
    }

    public float[,] Forward(float[,] X)
    {
        // 1) MHA + residual + norm
        var attnOut = Mha.Forward(X);
        var add1 = Add(X, attnOut);
        var norm1 = LayerNorm(add1);

        // 2) FFN + residual + norm
        var ffnOut = Ffn.Forward(norm1);
        var add2 = Add(norm1, ffnOut);
        var norm2 = LayerNorm(add2);

        return norm2;
    }

    private float[,] Add(float[,] A, float[,] B)
    {
        int r = A.GetLength(0), c = A.GetLength(1);
        var R = new float[r, c];
        for (var i = 0; i < r; i++)
        for (var j = 0; j < c; j++)
            R[i, j] = A[i, j] + B[i, j];
        return R;
    }

    public (float[,] dX, float[,] dWq, float[,] dWk, float[,] dWv, float[,] dWo,
        float[,] dW1, float[,] dW2) Backward(float[,] X, float[,] dY)
    {
        var attnOut = Mha.Forward(X);
        var add1 = Add(X, attnOut);
        var norm1 = LayerNorm(add1);

        var ffnOut = Ffn.Forward(norm1);
        var add2 = Add(norm1, ffnOut);
        var norm2 = LayerNorm(add2);

        var dAdd2 = LayerNormBackward(add2, dY);

        var dNorm1_from_res = dAdd2;
        var (dNorm1_from_ffn_in, dW1, dW2) = Ffn.Backward(norm1, dAdd2);

        var dNorm1 = Add(dNorm1_from_res, dNorm1_from_ffn_in);

        var dAdd1 = LayerNormBackward(add1, dNorm1);

        var dX_residual = dAdd1;
        var (dX_from_mha, dWq, dWk, dWv, dWo) = Mha.Backward(X, dAdd1);

        var dX = Add(dX_residual, dX_from_mha);

        return (dX, dWq, dWk, dWv, dWo, dW1, dW2);
    }

    private float[,] LayerNorm(float[,] X)
    {
        int r = X.GetLength(0), c = X.GetLength(1);
        var R = new float[r, c];

        for (var i = 0; i < r; i++)
        {
            var mean = 0f;
            for (var j = 0; j < c; j++) mean += X[i, j];
            mean /= c;

            var var = 0f;
            for (var j = 0; j < c; j++)
            {
                var z = X[i, j] - mean;
                var += z * z;
            }

            var /= c;

            var inv = 1.0f / MathF.Sqrt(var + 1e-5f);
            for (var j = 0; j < c; j++) R[i, j] = (X[i, j] - mean) * inv;
        }

        return R;
    }

    private float[,] LayerNormBackward(float[,] X, float[,] dY)
    {
        int r = X.GetLength(0), c = X.GetLength(1);
        var dX = new float[r, c];

        for (var i = 0; i < r; i++)
        {
            var mean = 0f;
            for (var j = 0; j < c; j++) mean += X[i, j];
            mean /= c;

            var var = 0f;
            for (var j = 0; j < c; j++)
            {
                var z = X[i, j] - mean;
                var += z * z;
            }

            var /= c;

            var inv = 1.0f / MathF.Sqrt(var + 1e-5f);

            float sum_dy = 0f, sum_dy_xhat = 0f;
            for (var j = 0; j < c; j++)
            {
                var xhat = (X[i, j] - mean) * inv;
                sum_dy += dY[i, j];
                sum_dy_xhat += dY[i, j] * xhat;
            }

            for (var j = 0; j < c; j++)
            {
                var xhat = (X[i, j] - mean) * inv;
                // standard formula: dX = (1/N)*inv*(N*dY - sum(dY) - xhat*sum(dY*xhat))
                dX[i, j] = 1f / c * inv * (c * dY[i, j] - sum_dy - xhat * sum_dy_xhat);
            }
        }

        return dX;
    }
}