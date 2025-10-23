namespace DaraGPT;

[Serializable]
public class Linear
{
    private static Random rng = new(1337);
    public float[] GradW, GradB;
    public int In, Out;
    public float[] W, B;

    public Linear(int inFeatures, int outFeatures)
    {
        In = inFeatures;
        Out = outFeatures;
        W = new float[Out * In];
        B = new float[Out];
        GradW = new float[Out * In];
        GradB = new float[Out];

        // Jača inicijalizacija da se signal ne ugasi kroz više slojeva
        for (var i = 0; i < W.Length; i++) W[i] = (float)((rng.NextDouble() - 0.5) * 0.1);
        for (var i = 0; i < B.Length; i++) B[i] = 0f;
    }

    public float[] Forward(float[] input, int batch, GpuEngine gpu = null)
    {
        var outArr = gpu != null && gpu.Available
            ? gpu.MatMul(input, W, batch, In, Out)
            : LinAlgCPU.MatMul(input, W, batch, In, Out);

        // + bias
        for (var i = 0; i < batch; i++)
        for (var j = 0; j < Out; j++)
            outArr[i * Out + j] += B[j];

        return outArr;
    }

    public float[] Backward(float[] input, float[] gradOutput, int batch)
    {
        Array.Clear(GradW, 0, GradW.Length);
        Array.Clear(GradB, 0, GradB.Length);

        // dW, dB
        for (var i = 0; i < Out; i++)
        {
            for (var j = 0; j < In; j++)
            {
                float sum = 0;
                for (var b = 0; b < batch; b++) sum += gradOutput[b * Out + i] * input[b * In + j];
                GradW[i * In + j] = sum / Math.Max(1, batch);
            }

            float sumB = 0;
            for (var b = 0; b < batch; b++) sumB += gradOutput[b * Out + i];
            GradB[i] = sumB / Math.Max(1, batch);
        }

        // dInput = gradOutput * W^T
        var gradInput = new float[batch * In];
        for (var b = 0; b < batch; b++)
        for (var j = 0; j < In; j++)
        {
            float sum = 0;
            for (var i = 0; i < Out; i++)
                sum += gradOutput[b * Out + i] * W[i * In + j];
            gradInput[b * In + j] = sum;
        }

        return gradInput;
    }

    public void ZeroGrad()
    {
        Array.Clear(GradW, 0, GradW.Length);
        Array.Clear(GradB, 0, GradB.Length);
    }
}

public class SGDOptimizer
{
    private readonly GpuEngine gpu;
    private readonly float lr;

    public SGDOptimizer(float learningRate, GpuEngine gpuEngine = null)
    {
        lr = learningRate;
        gpu = gpuEngine; // može biti null
    }

    public void Step(Linear layer)
    {
        if (gpu != null && gpu.Available)
        {
            gpu.UpdateWeights(layer.W, layer.GradW, lr);
            gpu.UpdateWeights(layer.B, layer.GradB, lr);
        }
        else
        {
            for (var i = 0; i < layer.W.Length; i++)
                layer.W[i] -= lr * layer.GradW[i];
            for (var i = 0; i < layer.B.Length; i++)
                layer.B[i] -= lr * layer.GradB[i];
        }
    }
}

public static class Utils
{
    public static float RandFloat()
    {
        return (float)new Random().NextDouble();
    }
}

public static class Activations
{
    public static float Gelu(float x)
    {
        return 0.5f * x * (1 + (float)Math.Tanh(0.79788456f * (x + 0.044715f * x * x * x)));
    }

    public static float GeluGrad(float x)
    {
        var tanh = (float)Math.Tanh(0.79788456f * (x + 0.044715f * x * x * x));
        var sech2 = 1 - tanh * tanh;
        return 0.5f * (1 + tanh + x * 0.79788456f * sech2 * (1 + 3 * 0.044715f * x * x));
    }
}