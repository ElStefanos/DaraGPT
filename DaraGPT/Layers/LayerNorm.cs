namespace DaraGPT;

[Serializable]
public class LayerNorm
{
    private readonly float[] beta;
    private readonly int dim;
    private readonly float eps = 1e-5f;
    private readonly float[] gamma;

    public LayerNorm(int dim)
    {
        this.dim = dim;
        gamma = new float[dim];
        beta = new float[dim];
        for (var i = 0; i < dim; i++)
        {
            gamma[i] = 1f;
            beta[i] = 0f;
        }
    }

    public float[] Forward(float[] x, int seqLen)
    {
        var y = new float[x.Length];
        for (var t = 0; t < seqLen; t++)
        {
            var mean = 0f;
            for (var d = 0; d < dim; d++) mean += x[t * dim + d];
            mean /= dim;

            var var = 0f;
            for (var d = 0; d < dim; d++)
            {
                var diff = x[t * dim + d] - mean;
                var += diff * diff;
            }

            var /= dim;
            var invStd = 1f / MathF.Sqrt(var + eps);

            for (var d = 0; d < dim; d++)
            {
                var idx = t * dim + d;
                y[idx] = gamma[d] * (x[idx] - mean) * invStd + beta[d];
            }
        }

        return y;
    }
}