using DaraGPT.Math;
using DaraGPT.Math.Gpu;

namespace DaraGPT.Model;

public class MultiHeadAttention
{
    private readonly int DHead;
    private readonly int DModel;
    private readonly GpuEngine Gpu;
    private readonly int NumHeads;
    private readonly RotaryEmbedding RoPe;
    public bool gpuWeightsReady;

    public float[,] Wq, Wk, Wv, Wo;
    public float[]? WqF, WkF, WvF, WoF;

    public MultiHeadAttention(int dModel, int numHeads, GpuEngine gpu = null)
    {
        if (dModel % numHeads != 0)
            throw new ArgumentException("dModel must be divisible by numHeads.");

        DModel = dModel;
        NumHeads = numHeads;
        DHead = dModel / numHeads;
        Gpu = gpu;
        RoPe = new RotaryEmbedding(DHead, gpu);

        Wq = InitWeights(dModel, dModel);
        Wk = InitWeights(dModel, dModel);
        Wv = InitWeights(dModel, dModel);
        Wo = InitWeights(dModel, dModel);

        if (gpu != null && gpu.Available)
            PrepareGpuWeights();
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

    private void PrepareGpuWeights()
    {
        WqF = MatrixUtils.ToFlat(Wq);
        WkF = MatrixUtils.ToFlat(Wk);
        WvF = MatrixUtils.ToFlat(Wv);
        WoF = MatrixUtils.ToFlat(Wo);
        gpuWeightsReady = true;
    }

    public float[,] Forward(float[,] X)
    {
        if (Gpu?.Available == true && gpuWeightsReady)
            try
            {
                return ForwardGpu(X);
            }
            catch (Exception ex)
            {
                Console.WriteLine("[MHA] GPU failed, fallback to CPU: " + ex.Message);
            }

        var seqLen = X.GetLength(0);

        var Q = MatrixUtils.MatMul(X, Wq);
        var K = MatrixUtils.MatMul(X, Wk);
        var V = MatrixUtils.MatMul(X, Wv);

        Q = SplitHeads(Q, seqLen);
        K = SplitHeads(K, seqLen);
        V = SplitHeads(V, seqLen);

        Q = RoPe.Apply(Q);
        K = RoPe.Apply(K);

        var scores = MatrixUtils.MatMul(Q, MatrixUtils.Transpose(K));
        MatrixUtils.ScaleInPlace(scores, 1.0f / MathF.Sqrt(DHead));
        var attention = MatrixUtils.Softmax(scores);

        var weighted = MatrixUtils.MatMul(attention, V);
        var output = MergeHeads(weighted, seqLen);
        return MatrixUtils.MatMul(output, Wo);
    }

    private float[,] ForwardGpu(float[,] X)
    {
        var seqLen = X.GetLength(0);
        var XF = MatrixUtils.ToFlat(X);

        var QF = OpenClKernels.RunMatMulRM(Gpu, XF, WqF!, seqLen, DModel, DModel);
        var KF = OpenClKernels.RunMatMulRM(Gpu, XF, WkF!, seqLen, DModel, DModel);
        var VF = OpenClKernels.RunMatMulRM(Gpu, XF, WvF!, seqLen, DModel, DModel);

        var Qh = SplitHeadsFlat(QF, seqLen);
        var Kh = SplitHeadsFlat(KF, seqLen);
        var Vh = SplitHeadsFlat(VF, seqLen);

        RoPe.ApplyGpu(Qh, NumHeads * seqLen);
        RoPe.ApplyGpu(Kh, NumHeads * seqLen);

        var KT = OpenClKernels.RunTranspose2D(Gpu, Kh, NumHeads * seqLen, DHead);
        var scores = OpenClKernels.RunMatMulRM(Gpu, Qh, KT, NumHeads * seqLen, DHead, NumHeads * seqLen);

        OpenClKernels.RunScale(Gpu, scores, 1.0f / MathF.Sqrt(DHead), NumHeads * seqLen, NumHeads * seqLen);
        var attn = OpenClKernels.RunRowSoftmax(Gpu, scores, NumHeads * seqLen, NumHeads * seqLen);

        var weighted = OpenClKernels.RunMatMulRM(Gpu, attn, Vh, NumHeads * seqLen, NumHeads * seqLen, DHead);

        var merged = MergeHeadsFlat(weighted, seqLen);
        var finalF = OpenClKernels.RunMatMulRM(Gpu, merged, WoF!, seqLen, DModel, DModel);
        return MatrixUtils.To2D(finalF, seqLen, DModel);
    }

    // CPU helpers
    public (float[,] dX, float[,] dWq, float[,] dWk, float[,] dWv, float[,] dWo) Backward(float[,] X, float[,] dOut)
    {
        if (Gpu?.Available == true && gpuWeightsReady)
            return BackwardGpu(X, dOut);


        var seqLen = X.GetLength(0);

        // Recompute forward intermediates
        var Q = MatrixUtils.MatMul(X, Wq);
        var K = MatrixUtils.MatMul(X, Wk);
        var V = MatrixUtils.MatMul(X, Wv);

        var Qh = SplitHeads(Q, seqLen);
        var Kh = SplitHeads(K, seqLen);
        var Vh = SplitHeads(V, seqLen);

        // RoPE forward
        Qh = RoPe.Apply(Qh);
        Kh = RoPe.Apply(Kh);

        var scale = 1.0f / MathF.Sqrt(DHead);
        var scores = MatrixUtils.MatMul(Qh, MatrixUtils.Transpose(Kh));
        MatrixUtils.ScaleInPlace(scores, scale);
        var A = MatrixUtils.Softmax(scores);
        var Z = MatrixUtils.MatMul(A, Vh); // [rows, dHead], rows = NumHeads*seqLen

        var merged = MergeHeads(Z, seqLen); // [seqLen, DModel]

        // 1) Grad: merged @ Wo -> dWo i dMerged
        var mergedT = MatrixUtils.Transpose(merged);
        var dWo = MatrixUtils.MatMul(mergedT, dOut); // [DModel, DModel]

        var WoT = MatrixUtils.Transpose(Wo);
        var dMerged = MatrixUtils.MatMul(dOut, WoT); // [seqLen, DModel]

        // 2) Splitting dMerged into heads (DIFFRENT FROM MERGEHEADS)
        var dZ = SplitHeads(dMerged, seqLen); // [rows, dHead]

        // 3) Z = A @ Vh
        var AT = MatrixUtils.Transpose(A);
        var dVh = MatrixUtils.MatMul(AT, dZ); // [rows, dHead]

        var ZT = MatrixUtils.Transpose(Z);
        var dA = MatrixUtils.MatMul(dZ, ZT); // [rows, rows]

        // 4) A = softmax(scores) po redovima → dScores
        var dScores = MatrixUtils.SoftmaxBackward(A, dA); // [rows, rows]
        MatrixUtils.ScaleInPlace(dScores, scale);

        // 5) scores = Qh @ Kh^T
        var KhT = MatrixUtils.Transpose(Kh);
        var dQh = MatrixUtils.MatMul(dScores, Kh); // [rows, dHead]
        var dKh = MatrixUtils.MatMul(MatrixUtils.Transpose(dScores), Qh); // [rows, dHead]

        // 6) Vh, Qh, Kh back to [seqLen, DModel]
        var dV = MergeHeads(dVh, seqLen); // [seqLen, DModel]
        var dQ = MergeHeads(dQh, seqLen);
        var dK = MergeHeads(dKh, seqLen);

        // 7) Q = X @ Wq; K = X @ Wk; V = X @ Wv
        var Xt = MatrixUtils.Transpose(X);
        var dWq = MatrixUtils.MatMul(Xt, dQ);
        var dWk = MatrixUtils.MatMul(Xt, dK);
        var dWv = MatrixUtils.MatMul(Xt, dV);

        var WqT = MatrixUtils.Transpose(Wq);
        var WkT = MatrixUtils.Transpose(Wk);
        var WvT = MatrixUtils.Transpose(Wv);

        var dXq = MatrixUtils.MatMul(dQ, WqT);
        var dXk = MatrixUtils.MatMul(dK, WkT);
        var dXv = MatrixUtils.MatMul(dV, WvT);

        // total grad on X: from all 3 times
        var dX = MatrixUtils.Add3(dXq, dXk, dXv);

        return (dX, dWq, dWk, dWv, dWo);
    }


    private (float[,] dX, float[,] dWq, float[,] dWk, float[,] dWv, float[,] dWo) BackwardGpu(float[,] X,
        float[,] dOut)
    {
        var seqLen = X.GetLength(0);

        // 1) Recompute Q,K,V on GPU
        var XF = MatrixUtils.ToFlat(X);
        var QF = OpenClKernels.RunMatMulRM(Gpu, XF, WqF!, seqLen, DModel, DModel);
        var KF = OpenClKernels.RunMatMulRM(Gpu, XF, WkF!, seqLen, DModel, DModel);
        var VF = OpenClKernels.RunMatMulRM(Gpu, XF, WvF!, seqLen, DModel, DModel);

        var Qh = SplitHeadsFlat(QF, seqLen);
        var Kh = SplitHeadsFlat(KF, seqLen);
        var Vh = SplitHeadsFlat(VF, seqLen);

        // RoPE forward (no grad w.r.t. angles for simplicity – kao CPU)
        RoPe.ApplyGpu(Qh, NumHeads * seqLen);
        RoPe.ApplyGpu(Kh, NumHeads * seqLen);

        // scores = Qh @ Kh^T
        var KT = OpenClKernels.RunTranspose2D(Gpu, Kh, NumHeads * seqLen, DHead);
        var scores = OpenClKernels.RunMatMulRM(Gpu, Qh, KT, NumHeads * seqLen, DHead, NumHeads * seqLen);

        // attn = softmax(scores/√d)
        var scale = 1.0f / MathF.Sqrt(DHead);
        OpenClKernels.RunScale(Gpu, scores, scale, NumHeads * seqLen, NumHeads * seqLen);
        var A = OpenClKernels.RunRowSoftmax(Gpu, scores, NumHeads * seqLen, NumHeads * seqLen);

        // Z = A @ Vh
        var Z = OpenClKernels.RunMatMulRM(Gpu, A, Vh, NumHeads * seqLen, NumHeads * seqLen, DHead);

        // merged = MergeHeads(Z)
        var merged = MergeHeadsFlat(Z, seqLen);

        // dWo = merged^T @ dOut
        var mergedT = OpenClKernels.RunTranspose2D(Gpu, merged, seqLen, DModel);
        var dWoF = OpenClKernels.RunMatMulRM(Gpu, mergedT, MatrixUtils.ToFlat(dOut), DModel, seqLen, DModel);
        var dWo = MatrixUtils.To2D(dWoF, DModel, DModel);

        // dMerged = dOut @ Wo^T
        var WT = OpenClKernels.RunTranspose2D(Gpu, WoF!, DModel, DModel);
        var dMerged = OpenClKernels.RunMatMulRM(Gpu, MatrixUtils.ToFlat(dOut), WT, seqLen, DModel, DModel);

        // dZ = SplitHeads(dMerged)
        var dZ = SplitHeadsFlat(dMerged, seqLen);

        // dVh = A^T @ dZ
        var AT = OpenClKernels.RunTranspose2D(Gpu, A, NumHeads * seqLen, NumHeads * seqLen);
        var dVh = OpenClKernels.RunMatMulRM(Gpu, AT, dZ, NumHeads * seqLen, NumHeads * seqLen, DHead);

        // dA = dZ @ Z^T
        var ZT = OpenClKernels.RunTranspose2D(Gpu, Z, NumHeads * seqLen, DHead);
        var dA = OpenClKernels.RunMatMulRM(Gpu, dZ, ZT, NumHeads * seqLen, DHead, NumHeads * seqLen);

        // dScores = SoftmaxBackward(A, dA) * (1/√d)
        var dScores = OpenClKernels.RunRowSoftmaxBackward(Gpu, A, dA, NumHeads * seqLen, NumHeads * seqLen);
        OpenClKernels.RunScale(Gpu, dScores, scale, NumHeads * seqLen, NumHeads * seqLen);

        // dQh = dScores @ Kh
        var dQh = OpenClKernels.RunMatMulRM(Gpu, dScores, Kh, NumHeads * seqLen, NumHeads * seqLen, DHead);

        // dKh = dScores^T @ Qh
        var dScoresT = OpenClKernels.RunTranspose2D(Gpu, dScores, NumHeads * seqLen, NumHeads * seqLen);
        var dKh = OpenClKernels.RunMatMulRM(Gpu, dScoresT, Qh, NumHeads * seqLen, NumHeads * seqLen, DHead);

        // Merge back to [seqLen, DModel]
        var dV = MergeHeadsFlat(dVh, seqLen);
        var dQ = MergeHeadsFlat(dQh, seqLen);
        var dK = MergeHeadsFlat(dKh, seqLen);

        // dWq = X^T @ dQ; etc.
        var Xt = OpenClKernels.RunTranspose2D(Gpu, XF, seqLen, DModel);
        var dWqF = OpenClKernels.RunMatMulRM(Gpu, Xt, dQ, DModel, seqLen, DModel);
        var dWkF = OpenClKernels.RunMatMulRM(Gpu, Xt, dK, DModel, seqLen, DModel);
        var dWvF = OpenClKernels.RunMatMulRM(Gpu, Xt, dV, DModel, seqLen, DModel);

        var dWq = MatrixUtils.To2D(dWqF, DModel, DModel);
        var dWk = MatrixUtils.To2D(dWkF, DModel, DModel);
        var dWv = MatrixUtils.To2D(dWvF, DModel, DModel);

        // dX = dQ @ Wq^T + dK @ Wk^T + dV @ Wv^T
        var WqT = OpenClKernels.RunTranspose2D(Gpu, WqF!, DModel, DModel);
        var WkT = OpenClKernels.RunTranspose2D(Gpu, WkF!, DModel, DModel);
        var WvT = OpenClKernels.RunTranspose2D(Gpu, WvF!, DModel, DModel);

        var dXqF = OpenClKernels.RunMatMulRM(Gpu, dQ, WqT, seqLen, DModel, DModel);
        var dXkF = OpenClKernels.RunMatMulRM(Gpu, dK, WkT, seqLen, DModel, DModel);
        var dXvF = OpenClKernels.RunMatMulRM(Gpu, dV, WvT, seqLen, DModel, DModel);

        var dXq = MatrixUtils.To2D(dXqF, seqLen, DModel);
        var dXk = MatrixUtils.To2D(dXkF, seqLen, DModel);
        var dXv = MatrixUtils.To2D(dXvF, seqLen, DModel);

        var dX = MatrixUtils.Add3(dXq, dXk, dXv); // same as CPU:contentReference[oaicite:7]{index=7}

        return (dX, dWq, dWk, dWv, dWo);
    }


    // helpers
    private float[,] SplitHeads(float[,] X, int seqLen)
    {
        var result = new float[NumHeads * seqLen, DHead];
        for (var h = 0; h < NumHeads; h++)
        for (var i = 0; i < seqLen; i++)
        for (var j = 0; j < DHead; j++)
            result[h * seqLen + i, j] = X[i, h * DHead + j];
        return result;
    }

    private float[,] MergeHeads(float[,] X, int seqLen)
    {
        var result = new float[seqLen, DModel];
        for (var h = 0; h < NumHeads; h++)
        for (var i = 0; i < seqLen; i++)
        for (var j = 0; j < DHead; j++)
            result[i, h * DHead + j] = X[h * seqLen + i, j];
        return result;
    }

    private float[] SplitHeadsFlat(float[] flat, int seqLen)
    {
        var outF = new float[NumHeads * seqLen * DHead];
        for (var h = 0; h < NumHeads; h++)
        for (var i = 0; i < seqLen; i++)
        for (var j = 0; j < DHead; j++)
            outF[(h * seqLen + i) * DHead + j] = flat[i * DModel + h * DHead + j];
        return outF;
    }

    private float[] MergeHeadsFlat(float[] flatHeads, int seqLen)
    {
        var outF = new float[seqLen * DModel];
        for (var h = 0; h < NumHeads; h++)
        for (var i = 0; i < seqLen; i++)
        for (var j = 0; j < DHead; j++)
            outF[i * DModel + h * DHead + j] = flatHeads[(h * seqLen + i) * DHead + j];
        return outF;
    }
}