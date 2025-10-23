using System;

namespace DaraGPT
{
    [Serializable]
    public class LayerNorm
    {
        private readonly int dim;
        private readonly float eps = 1e-5f;
        private readonly float[] gamma;
        private readonly float[] beta;

        public LayerNorm(int dim)
        {
            this.dim = dim;
            gamma = new float[dim];
            beta = new float[dim];
            for (int i = 0; i < dim; i++)
            {
                gamma[i] = 1f;
                beta[i] = 0f;
            }
        }

        public float[] Forward(float[] x, int seqLen)
        {
            var y = new float[x.Length];
            for (int t = 0; t < seqLen; t++)
            {
                float mean = 0f;
                for (int d = 0; d < dim; d++) mean += x[t * dim + d];
                mean /= dim;

                float var = 0f;
                for (int d = 0; d < dim; d++)
                {
                    float diff = x[t * dim + d] - mean;
                    var += diff * diff;
                }
                var /= dim;
                float invStd = 1f / MathF.Sqrt(var + eps);

                for (int d = 0; d < dim; d++)
                {
                    int idx = t * dim + d;
                    y[idx] = gamma[d] * (x[idx] - mean) * invStd + beta[d];
                }
            }
            return y;
        }
    }
}