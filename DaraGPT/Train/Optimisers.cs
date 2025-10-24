namespace DaraGPT.Train;

public class SGD
{
    public SGD(float lr)
    {
        LearningRate = lr;
    }

    public float LearningRate { get; }

    public void Step(float[] w, float[] dw)
    {
        for (var i = 0; i < w.Length; i++)
            w[i] -= LearningRate * dw[i];
    }

    public void Step(float[,] W, float[,] dW)
    {
        int r = W.GetLength(0), c = W.GetLength(1);
        for (var i = 0; i < r; i++)
        for (var j = 0; j < c; j++)
            W[i, j] -= LearningRate * dW[i, j];
    }
}