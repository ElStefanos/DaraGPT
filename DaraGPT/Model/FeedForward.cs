using DaraGPT.Math;

namespace DaraGPT.Model;

public class FeedForward
{
    private readonly int dModel;
    private readonly int hidden;
    public float[,] W1;
    public float[,] W2;

    public FeedForward(int dModel, int hidden)
    {
        this.dModel = dModel;
        this.hidden = hidden;

        W1 = InitWeights(dModel, hidden);
        W2 = InitWeights(hidden, dModel);
    }

    private float[,] InitWeights(int rows, int cols)
    {
        var rnd = new Random();
        var m = new float[rows, cols];
        for (var i = 0; i < rows; i++)
        for (var j = 0; j < cols; j++)
            m[i, j] = (float)(rnd.NextDouble() * 0.02 - 0.01);
        return m;
    }

    public float[,] Forward(float[,] X)
    {
        var H = MatrixUtils.MatMul(X, W1);
        ApplyReLU(H);
        return MatrixUtils.MatMul(H, W2);
    }

    public (float[,] dX, float[,] dW1, float[,] dW2) Backward(float[,] X, float[,] dOut)
    {
        // Recompute hidden pre-activation and mask
        var H = MatrixUtils.MatMul(X, W1); // [seqLen, hidden]

        // dW2 = H^T @ dOut
        var Ht = MatrixUtils.Transpose(H);
        var dW2 = MatrixUtils.MatMul(Ht, dOut);

        // dH = dOut @ W2^T
        var W2t = MatrixUtils.Transpose(W2);
        var dH = MatrixUtils.MatMul(dOut, W2t);

        // ReLU backward: dH *= (H > 0)
        int r = H.GetLength(0), c = H.GetLength(1);
        for (var i = 0; i < r; i++)
        for (var j = 0; j < c; j++)
            if (H[i, j] <= 0f)
                dH[i, j] = 0f;

        // dW1 = X^T @ dH
        var Xt = MatrixUtils.Transpose(X);
        var dW1 = MatrixUtils.MatMul(Xt, dH);

        // dX = dH @ W1^T
        var W1t = MatrixUtils.Transpose(W1);
        var dX = MatrixUtils.MatMul(dH, W1t);

        return (dX, dW1, dW2);
    }

    private void ApplyReLU(float[,] X)
    {
        var rows = X.GetLength(0);
        var cols = X.GetLength(1);
        for (var i = 0; i < rows; i++)
        for (var j = 0; j < cols; j++)
            X[i, j] = MathF.Max(0, X[i, j]);
    }
}