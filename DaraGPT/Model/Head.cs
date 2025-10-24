using DaraGPT.Math;
using DaraGPT.Math.Gpu;

namespace DaraGPT.Model;

public class Head
{
    private readonly int dModel;
    private readonly int vocabSize;

    private float[,] _lastDW;

    public Head(int vocabSize, int dModel)
    {
        this.vocabSize = vocabSize;
        this.dModel = dModel;
        Weights = InitWeights(dModel, vocabSize);
    }

    public float[,] Weights { get; set; }

    private float[,] InitWeights(int rows, int cols)
    {
        Random rnd = new();
        var w = new float[rows, cols];
        for (var i = 0; i < rows; i++)
        for (var j = 0; j < cols; j++)
            w[i, j] = (float)(rnd.NextDouble() * 0.02 - 0.01);
        return w;
    }

    public float[] Forward(float[,] X)
    {
        var out2D = MatrixUtils.MatMul(X, Weights); // [seqLen, vocab]
        return MatrixUtils.ToFlat(out2D);
    }

    // dLogits: [seqLen, vocab]
    public float[,] Backward(float[,] X, float[,] dLogits, GpuEngine gpu = null)
    {
        var seqLen = X.GetLength(0);
        var V = dLogits.GetLength(1);

        float[,] dW, dX;

        if (gpu != null && gpu.Available)
        {
            // GPU branch
            var XF = MatrixUtils.ToFlat(X);
            var dLF = MatrixUtils.ToFlat(dLogits);
            var WF = MatrixUtils.ToFlat(Weights);

            var XT = OpenClKernels.RunTranspose2D(gpu, XF, seqLen, dModel); // [dModel*seqLen]
            var dWF = OpenClKernels.RunMatMulRM(gpu, XT, dLF, dModel, seqLen, V); // [dModel*V]
            dW = MatrixUtils.To2D(dWF, dModel, V);

            var WT = OpenClKernels.RunTranspose2D(gpu, WF, dModel, V); // [V*dModel]
            var dXF = OpenClKernels.RunMatMulRM(gpu, dLF, WT, seqLen, V, dModel); // [seqLen*dModel]
            dX = MatrixUtils.To2D(dXF, seqLen, dModel);
        }
        else
        {
            // CPU branch
            var XT = MatrixUtils.Transpose(X);
            dW = MatrixUtils.MatMul(XT, dLogits);

            var WT = MatrixUtils.Transpose(Weights);
            dX = MatrixUtils.MatMul(dLogits, WT);
        }

        _lastDW = dW;
        return dX;
    }

    public float[,] ConsumeGradW()
    {
        var g = _lastDW;
        _lastDW = null!;
        return g;
    }
}