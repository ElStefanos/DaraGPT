namespace DaraGPT.Train;

public static class Loss
{
    public static float CrossEntropy(float[] logits, int targetId, out float[] dLogits)
    {
        var V = logits.Length;
        dLogits = new float[V];

        var max = float.NegativeInfinity;
        for (var i = 0; i < V; i++)
            if (logits[i] > max)
                max = logits[i];

        var sumExp = 0f;
        for (var i = 0; i < V; i++) sumExp += MathF.Exp(logits[i] - max);

        var logProb = logits[targetId] - max - MathF.Log(sumExp);
        var loss = -logProb;

        var invSum = 1f / sumExp;
        for (var i = 0; i < V; i++)
            dLogits[i] = MathF.Exp(logits[i] - max) * invSum;
        dLogits[targetId] -= 1f;

        return loss;
    }


    public static float CrossEntropySeqBatch(
        float[] logitsBTV, int B, int T, int V, int[][] targetsB,
        bool useOnlyLast, out float[] dLogitsBTV)
    {
        dLogitsBTV = new float[B * T * V];
        var total = 0f;
        var denom = useOnlyLast ? B : B * T;

        if (useOnlyLast)
            for (var b = 0; b < B; b++)
            {
                var t = T - 1;
                var off = (b * T + t) * V;

                var row = new float[V];
                Array.Copy(logitsBTV, off, row, 0, V);

                float[] dRow;
                total += CrossEntropy(row, targetsB[b][t], out dRow);
                Array.Copy(dRow, 0, dLogitsBTV, off, V);
            }
        else
            for (var b = 0; b < B; b++)
            for (var t = 0; t < T; t++)
            {
                var off = (b * T + t) * V;
                var row = new float[V];
                Array.Copy(logitsBTV, off, row, 0, V);

                float[] dRow;
                total += CrossEntropy(row, targetsB[b][t], out dRow);
                Array.Copy(dRow, 0, dLogitsBTV, off, V);
            }

        return total / denom;
    }


    public static float CrossEntropySeq(float[] logits2D, int seqLen, int vocab, int[] targetIds,
        bool useOnlyLast, out float[] dLogits2D)
    {
        dLogits2D = new float[seqLen * vocab];
        var total = 0f;
        var count = useOnlyLast ? 1 : seqLen;

        if (useOnlyLast)
        {
            var t = seqLen - 1;
            var off = t * vocab;
            var row = new float[vocab];
            Array.Copy(logits2D, off, row, 0, vocab);
            float[] dRow;
            total += CrossEntropy(row, targetIds[t], out dRow);
            Array.Copy(dRow, 0, dLogits2D, off, vocab);
        }
        else
        {
            for (var t = 0; t < seqLen; t++)
            {
                var off = t * vocab;
                var row = new float[vocab];
                Array.Copy(logits2D, off, row, 0, vocab);
                float[] dRow;
                total += CrossEntropy(row, targetIds[t], out dRow);
                Array.Copy(dRow, 0, dLogits2D, off, vocab);
            }
        }

        return total / count;
    }
}