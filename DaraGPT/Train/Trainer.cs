using DaraGPT.Math;
using DaraGPT.Math.Gpu;
using DaraGPT.Model;

namespace DaraGPT.Train;

public class Trainer
{
    private readonly GpuEngine gpu;
    private readonly GPTModel model;
    private readonly SGD sgd;

    public Trainer(GPTModel model, float learningRate, GpuEngine gpu = null)
    {
        this.model = model;
        sgd = new SGD(learningRate);
        this.gpu = gpu;
    }
    
    public float TrainBatch(IReadOnlyList<int[]> batchTokens, bool lossOnlyLast = true)
    {
        var B = batchTokens.Count;
        if (B == 0) return 0f;
        var T = batchTokens[0].Length;
        var D = model.DModel;

        var inputBT = new int[B * T];
        var targetsB = new int[B][];
        for (var b = 0; b < B; b++)
        {
            var seq = batchTokens[b];
            if (seq.Length != T)
                throw new Exception("Sve sekvence u batch-u moraju imati istu dužinu T.");

            targetsB[b] = new int[T];
            for (var t = 0; t < T; t++)
            {
                inputBT[b * T + t] = seq[t];
                targetsB[b][t] = t + 1 < T ? seq[t + 1] : seq[t];
            }
        }

        // 2) Forward through Embedding
        var embFlatBTD = model.Embedding.Forward(inputBT); // [B*T * D]
        var X = MatrixUtils.To2D(embFlatBTD, B * T, D);

        // 3) Forward through TransformerBlocks (caching X before each block for backward)
        var XinputsPerBlock = new List<float[,]>(model.Blocks.Length);
        for (var i = 0; i < model.Blocks.Length; i++)
        {
            XinputsPerBlock.Add(X);
            X = model.Blocks[i].Forward(X);
        }

        // 4) Final LayerNorm (Due to OpenCL limitation it runs on CPU)
        X = FinalLayerNorm(X);

        // 5) Head → logits
        var logitsFlat = model.Head.Forward(X); // [B*T * V]
        var V = logitsFlat.Length / (B * T);

        // 6) Loss (+ dLogits po B,T redu)
        float[] dLogitsFlat;
        var loss = Loss.CrossEntropySeqBatch(logitsFlat, B, T, V, targetsB, lossOnlyLast, out dLogitsFlat);

        // 7) Head backward
        var dLogits = MatrixUtils.To2D(dLogitsFlat, B * T, V);
        var dX = model.Head.Backward(X, dLogits, gpu);
        var dW_head = model.Head.ConsumeGradW();

        // 8) Backward through blokove (reverse order)
        for (var i = model.Blocks.Length - 1; i >= 0; i--)
        {
            // VAŽNO: prosleđuje se X iz forward-a za taj blok + dY u taj blok
            var (dXin, dWq, dWk, dWv, dWo, dW1, dW2) = model.Blocks[i].Backward(XinputsPerBlock[i], dX);

            // SGD update on block weights
            sgd.Step(model.Blocks[i].Mha.Wq, dWq);
            sgd.Step(model.Blocks[i].Mha.Wk, dWk);
            sgd.Step(model.Blocks[i].Mha.Wv, dWv);
            sgd.Step(model.Blocks[i].Mha.Wo, dWo);
            sgd.Step(model.Blocks[i].FfnW1, dW1);
            sgd.Step(model.Blocks[i].FfnW2, dW2);

            dX = dXin;
        }

        // 9) Embedding grad (scatter-add)
        var dE = model.Embedding.BackwardAccumulate(inputBT, dX);
        model.Embedding.ApplyGradients(dE, sgd.LearningRate);

        // 10) Head update
        sgd.Step(model.Head.Weights, dW_head);

        return loss;
    }

    /// <summary>
    ///     Epoch loop
    /// </summary>
    public void TrainEpoch(IReadOnlyList<int[]> allSequences, int batchSize, bool lossOnlyLast = true,
        Action<int, float> onBatch = null)
    {
        var N = allSequences.Count;
        int idx = 0, b = 0;
        while (idx < N)
        {
            var batch = new List<int[]>(batchSize);
            for (var i = 0; i < batchSize && idx < N; i++, idx++)
                batch.Add(allSequences[idx]);

            var loss = TrainBatch(batch, lossOnlyLast);
            onBatch?.Invoke(++b, loss);
        }
    }

    private float[,] FinalLayerNorm(float[,] X)
    {
        var rows = X.GetLength(0);
        var cols = X.GetLength(1);
        var R = new float[rows, cols];

        for (var i = 0; i < rows; i++)
        {
            var mean = 0f;
            for (var j = 0; j < cols; j++) mean += X[i, j];
            mean /= cols;

            var var = 0f;
            for (var j = 0; j < cols; j++)
            {
                var z = X[i, j] - mean;
                var += z * z;
            }

            var /= cols;

            var inv = 1.0f / MathF.Sqrt(var + 1e-5f);
            for (var j = 0; j < cols; j++)
                R[i, j] = (X[i, j] - mean) * inv;
        }

        return R;
    }
}