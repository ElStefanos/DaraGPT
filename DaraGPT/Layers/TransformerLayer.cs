namespace DaraGPT;

[Serializable]
public class TransformerLayer
{
    private readonly GpuEngine gpu;

    private readonly LayerNorm ln1;
    private readonly LayerNorm ln2;
    public int DModel;
    public Linear Wq, Wk, Wv, Wo, MLP1, MLP2;


    public TransformerLayer(int dModel, GpuEngine gpuEngine = null)
    {
        DModel = dModel;
        gpu = gpuEngine;
        Wq = new Linear(DModel, DModel);
        Wk = new Linear(DModel, DModel);
        Wv = new Linear(DModel, DModel);
        Wo = new Linear(DModel, DModel);
        MLP1 = new Linear(DModel, DModel * 4);
        MLP2 = new Linear(DModel * 4, DModel);

        ln1 = new LayerNorm(DModel);
        ln2 = new LayerNorm(DModel);
    }

    public float[] Forward(float[] x, int seqLen)
    {
        //1. LayerNorm pre attention-a
        var norm1 = ln1.Forward(x, seqLen);

        var q = Wq.Forward(norm1, seqLen, gpu);
        var k = Wk.Forward(norm1, seqLen, gpu);
        var v = Wv.Forward(norm1, seqLen, gpu);

        // Transponuj K
        var kT = new float[DModel * seqLen];
        for (var r = 0; r < seqLen; r++)
        for (var c = 0; c < DModel; c++)
            kT[c * seqLen + r] = k[r * DModel + c];

        // Attention score
        var scores = gpu != null && gpu.Available
            ? gpu.MatMul(q, kT, seqLen, DModel, seqLen)
            : LinAlgCPU.MatMul(q, kT, seqLen, DModel, seqLen);

        // Skaliranje po d_model
        var scale = 1.0f / MathF.Sqrt(DModel);
        for (var i = 0; i < scores.Length; i++)
            scores[i] *= scale;

        // Softmax
        var attn = gpu != null && gpu.Available
            ? gpu.RowSoftmax(scores, seqLen, seqLen)
            : LinAlgCPU.SoftmaxRowwise(scores, seqLen, seqLen);

        // Attention * V
        var outVec = gpu != null && gpu.Available
            ? gpu.MatMul(attn, v, seqLen, seqLen, DModel)
            : LinAlgCPU.MatMul(attn, v, seqLen, seqLen, DModel);

        // Prolaz kroz linearni izlazni sloj
        var projected = Wo.Forward(outVec, seqLen, gpu);

        // Dropout (10%)
        for (var i = 0; i < projected.Length; i++)
            if (Utils.RandFloat() < 0.1f)
                projected[i] = 0;

        // Residual scaling 0.9
        var resid1 = new float[projected.Length];
        for (var i = 0; i < resid1.Length; i++)
            resid1[i] = x[i] + 0.9f * projected[i];

        // 2. LayerNorm pre MLP-a ===
        var norm2 = ln2.Forward(resid1, seqLen);

        // MLP blok
        var hid = MLP1.Forward(norm2, seqLen, gpu);
        for (var i = 0; i < hid.Length; i++)
            hid[i] = Activations.Gelu(hid[i]);
        var mlpOut = MLP2.Forward(hid, seqLen, gpu);

        // Dropout u MLP izlazu
        for (var i = 0; i < mlpOut.Length; i++)
            if (Utils.RandFloat() < 0.1f)
                mlpOut[i] = 0;

        // Final residual skip sa scalingom
        var final = new float[mlpOut.Length];
        for (var i = 0; i < final.Length; i++)
            final[i] = resid1[i] + 0.9f * mlpOut[i];

        return final;
    }


    // Minimalni Backward (MLP + skip), ostalo je demo – održava kompatibilnost
    public float[] Backward(float[] gradOutput, float[] x, int seqLen, float lr)
    {
        var hid = MLP1.Forward(x, seqLen, gpu);
        for (var i = 0; i < hid.Length; i++) hid[i] = Activations.Gelu(hid[i]);

        var gradMlp = MLP2.Backward(hid, gradOutput, seqLen);
        var gradHid = new float[hid.Length];
        for (var i = 0; i < hid.Length; i++)
            gradHid[i] = gradMlp[i] * Activations.GeluGrad(hid[i]);

        var gradProjected = MLP1.Backward(x, gradHid, seqLen);

        var gradOutVec = Wo.Backward(x, gradOutput, seqLen);
        for (var i = 0; i < gradOutVec.Length; i++)
            gradOutVec[i] += gradProjected[i];

        Wq.ZeroGrad();
        Wk.ZeroGrad();
        Wv.ZeroGrad();
        Wo.ZeroGrad();
        MLP1.ZeroGrad();
        MLP2.ZeroGrad();

        var opt = new SGDOptimizer(lr, gpu);
        opt.Step(Wq);
        opt.Step(Wk);
        opt.Step(Wv);
        opt.Step(Wo);
        opt.Step(MLP1);
        opt.Step(MLP2);

        return gradOutVec;
    }
}