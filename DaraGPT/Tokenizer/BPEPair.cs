namespace DaraGPT.Tokenizer;

/**
 * Represents a pair of symbols and its frequency.
 * Used internally by BPETokenizer to determine which pairs to merge.
 */
public class BPEPair
{
    public BPEPair(string left, string right, int freq)
    {
        Symbols = (left, right);
        Frequency = freq;
    }

    public (string Left, string Right) Symbols { get; set; }
    public int Frequency { get; set; }

    public string __toString()
    {
        return $"Pair: {Symbols.Left} + {Symbols.Right} | Freq: {Frequency}";
    }
}