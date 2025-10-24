// DaraGPT/Model/Embedding.cs

using DaraGPT.Math.Gpu;
using OpenCL.Net;

namespace DaraGPT.Model;

public class Embedding : IDisposable
{
    private readonly GpuEngine Gpu;
    private readonly int VocabSize;
    public readonly float[] Weights;
    private bool gpuReady;
    private IMem<float>? gpuWeights;

    public Embedding(GpuEngine gpu, int vocabSize, int dModel)
    {
        VocabSize = vocabSize;
        Dim = dModel;
        Gpu = gpu;

        Weights = InitWeights(vocabSize, dModel);

        if (gpu != null && gpu.Available)
            UploadToGpu();
    }

    public int Dim { get; }


    public void Dispose()
    {
        if (gpuWeights != null)
        {
            Cl.ReleaseMemObject(gpuWeights);
            gpuReady = false;
        }
    }

    private float[] InitWeights(int rows, int cols)
    {
        Random rnd = new();
        var w = new float[rows * cols];
        for (var i = 0; i < w.Length; i++)
            w[i] = (float)(rnd.NextDouble() * 0.02 - 0.01);
        return w;
    }

    private void UploadToGpu()
    {
        gpuWeights = Cl.CreateBuffer(
            Gpu.Context,
            MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            Weights,
            out var err);

        if (err != ErrorCode.Success)
            throw new Exception("Failed to upload embedding weights to GPU.");

        gpuReady = true;
        Console.WriteLine($"[Embedding] Uploaded weights to GPU ({VocabSize} x {Dim}).");
    }

    public float[] Forward(int[] tokenIds)
    {
        if (Gpu != null && Gpu.Available && gpuReady)
            return OpenClKernels.RunEmbeddingLookup(Gpu, tokenIds, gpuWeights!, Dim);

        var output = new float[tokenIds.Length * Dim];
        for (var i = 0; i < tokenIds.Length; i++)
        {
            var t = tokenIds[i];
            var oBase = i * Dim;
            var wBase = t * Dim;
            for (var j = 0; j < Dim; j++)
                output[oBase + j] = Weights[wBase + j];
        }

        return output;
    }

    public void LoadWeights(float[] newWeights)
    {
        if (newWeights.Length != Weights.Length)
            throw new Exception($"Embedding weight size mismatch: expected {Weights.Length}, got {newWeights.Length}");
        Array.Copy(newWeights, Weights, Weights.Length);

        if (Gpu != null && Gpu.Available)
            UploadToGpu();
    }


    // dX: [seqLen, dModel]; tokenIds: [seqLen]
    public float[] BackwardAccumulate(int[] tokenIds, float[,] dX)
    {
        var seqLen = tokenIds.Length;
        var dE = new float[Weights.Length]; //[vocabSize * dModel]

        for (var i = 0; i < seqLen; i++)
        {
            var t = tokenIds[i];
            var eBase = t * Dim;
            for (var j = 0; j < Dim; j++)
                dE[eBase + j] += dX[i, j];
        }

        return dE;
    }

    public void ApplyGradients(float[] dE, float lr)
    {
        for (var i = 0; i < Weights.Length; i++)
            Weights[i] -= lr * dE[i];

        if (Gpu.Available) UploadToGpu();
    }
}