namespace DaraGPT.Model;

public class Predictor
{
    private readonly GPTModel model;

    public Predictor(GPTModel model)
    {
        this.model = model;
    }

    public int PredictNext(int[] context)
    {
        var logits = model.Forward(context);

        var seqLen = context.Length;
        var vocab = logits.Length / seqLen;

        var start = (seqLen - 1) * vocab;
        var best = 0;
        var max = logits[start];

        for (var i = 1; i < vocab; i++)
        {
            var v = logits[start + i];
            if (v > max)
            {
                max = v;
                best = i;
            }
        }

        return best;
    }
}