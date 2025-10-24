using DaraGPT.Math;
using DaraGPT.Math.Gpu;

namespace DaraGPT.Model;

public class GPTModel
{
    public readonly int DModel;
    public readonly Embedding Embedding;
    public readonly Head Head;
    public TransformerBlock[] Blocks;
    public Config.Config Config;

    public GPTModel(GpuEngine gpu, Config.Config cfg)
    {
        Config = cfg;
        DModel = cfg.DModel;

        Embedding = new Embedding(gpu, cfg.VocabSize, cfg.DModel);

        Blocks = new TransformerBlock[cfg.Layers];
        for (var i = 0; i < cfg.Layers; i++)
            Blocks[i] = new TransformerBlock(cfg.DModel, cfg.Head, gpu);

        Head = new Head(cfg.VocabSize, cfg.DModel);
    }

    public float[] Forward(int[] tokens)
    {
        var emb = Embedding.Forward(tokens);
        var X = MatrixUtils.To2D(emb, tokens.Length, DModel);

        for (var i = 0; i < Blocks.Length; i++)
            X = Blocks[i].Forward(X);

        X = FinalLayerNorm(X);
        return Head.Forward(X);
    }

    private float[,] FinalLayerNorm(float[,] X)
    {
        var rows = X.GetLength(0);
        var cols = X.GetLength(1);
        var R = new float[rows, cols];

        for (var i = 0; i < rows; i++)
        {
            var mean = 0f;
            for (var j = 0; j < cols; j++)
                mean += X[i, j];
            mean /= cols;

            var var = 0f;
            for (var j = 0; j < cols; j++)
                var += (X[i, j] - mean) * (X[i, j] - mean);
            var /= cols;

            var invStd = 1.0f / MathF.Sqrt(var + 1e-5f);
            for (var j = 0; j < cols; j++)
                R[i, j] = (X[i, j] - mean) * invStd;
        }

        return R;
    }
}