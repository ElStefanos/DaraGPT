namespace DaraGPT.Math;

public static class MatrixUtils
{
    public static float[,] MatMul(float[,] A, float[,] B)
    {
        var aRows = A.GetLength(0);
        var aCols = A.GetLength(1);
        var bCols = B.GetLength(1);

        var result = new float[aRows, bCols];
        for (var i = 0; i < aRows; i++)
        for (var k = 0; k < aCols; k++)
        {
            var aval = A[i, k];
            for (var j = 0; j < bCols; j++)
                result[i, j] += aval * B[k, j];
        }

        return result;
    }

    public static float[,] Transpose(float[,] A)
    {
        var rows = A.GetLength(0);
        var cols = A.GetLength(1);
        var T = new float[cols, rows];

        for (var i = 0; i < rows; i++)
        for (var j = 0; j < cols; j++)
            T[j, i] = A[i, j];
        return T;
    }

    public static void ScaleInPlace(float[,] A, float s)
    {
        var rows = A.GetLength(0);
        var cols = A.GetLength(1);
        for (var i = 0; i < rows; i++)
        for (var j = 0; j < cols; j++)
            A[i, j] *= s;
    }

    //Softmax (row-wise)
    public static float[,] Softmax(float[,] A)
    {
        var rows = A.GetLength(0);
        var cols = A.GetLength(1);
        var result = new float[rows, cols];

        for (var i = 0; i < rows; i++)
        {
            var max = float.NegativeInfinity;
            for (var j = 0; j < cols; j++)
                if (A[i, j] > max)
                    max = A[i, j];

            var sum = 0f;
            for (var j = 0; j < cols; j++)
                sum += MathF.Exp(A[i, j] - max);

            var inv = 1f / sum;
            for (var j = 0; j < cols; j++)
                result[i, j] = MathF.Exp(A[i, j] - max) * inv;
        }

        return result;
    }

    //Flatten / reshape 
    public static float[] ToFlat(float[,] A)
    {
        var rows = A.GetLength(0);
        var cols = A.GetLength(1);
        var f = new float[rows * cols];
        Buffer.BlockCopy(A, 0, f, 0, sizeof(float) * f.Length);
        return f;
    }

    public static float[,] To2D(float[] f, int rows, int cols)
    {
        var A = new float[rows, cols];
        Buffer.BlockCopy(f, 0, A, 0, sizeof(float) * f.Length);
        return A;
    }

    public static float[,] Add3(float[,] A, float[,] B, float[,] C)
    {
        int r = A.GetLength(0), c = A.GetLength(1);
        var R = new float[r, c];
        for (var i = 0; i < r; i++)
        for (var j = 0; j < c; j++)
            R[i, j] = A[i, j] + B[i, j] + C[i, j];
        return R;
    }

    // derivative of softmax given upstream gradient dY and softmax output Y
    // formula: dX = (dY - sum(dY * Y)) * Y

    public static float[,] SoftmaxBackward(float[,] Y, float[,] dY)
    {
        var r = Y.GetLength(0);
        var c = Y.GetLength(1);
        var dX = new float[r, c];

        for (var i = 0; i < r; i++)
        {
            var dot = 0f;
            for (var j = 0; j < c; j++)
                dot += dY[i, j] * Y[i, j];

            for (var j = 0; j < c; j++)
                dX[i, j] = (dY[i, j] - dot) * Y[i, j];
        }

        return dX;
    }
}