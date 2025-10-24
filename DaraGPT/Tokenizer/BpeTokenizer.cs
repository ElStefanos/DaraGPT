namespace DaraGPT.Tokenizer;

/**
 * Byte Pair Encoding (BPE) Tokenizer
 * Handles vocabulary training, encoding, decoding, and sequence creation.
 */
public class BpeTokenizer
{
    private const string Eow = "</w>";
    private readonly List<(string, string)> _merges = new();

    public BpeTokenizer()
    {
        AddToken("<PAD>");
        AddToken("<UNK>");
        AddToken("<EOS>");
    }

    public Dictionary<string, int> TokenToId { get; set; } = new();
    public Dictionary<int, string> IdToToken { get; set; } = new();

    public List<List<int>> Sequences { get; set; } = new();

    private int AddToken(string token)
    {
        if (TokenToId.ContainsKey(token))
            return TokenToId[token];

        var id = TokenToId.Count;
        TokenToId[token] = id;
        IdToToken[id] = token;
        return id;
    }

    /// <summary>
    ///     Trains the BPE tokenizer on a given text corpus.
    /// </summary>
    public void TrainBpe(IEnumerable<string> texts, int vocabTarget = 5000)
    {
        Console.WriteLine("Training BPE Tokenizer...");

        var vocab = new Dictionary<string, int>();

        foreach (var text in texts)
        foreach (var word in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
            var chars = word.Select(c => c.ToString()).ToList();
            chars.Add(Eow);
            var joined = string.Join(" ", chars);

            if (!vocab.ContainsKey(joined))
                vocab[joined] = 0;
            vocab[joined]++;
        }

        Console.WriteLine($"Initial vocab size: {vocab.Count}");

        while (TokenToId.Count < vocabTarget)
        {
            var pairs = CountPairs(vocab);
            if (pairs.Count == 0)
                break;

            var best = pairs.OrderByDescending(p => p.Frequency).First();
            var left = best.Symbols.Left;
            var right = best.Symbols.Right;
            var newSymbol = left + right;

            _merges.Add((left, right));
            AddToken(newSymbol);

            vocab = MergePair(vocab, left, right, newSymbol);
        }

        Console.WriteLine($"Training finished. Final vocab size: {TokenToId.Count}");
    }

    /// <summary>
    ///     Counts symbol pair frequencies and returns them as BPEPair objects.
    /// </summary>
    private List<BPEPair> CountPairs(Dictionary<string, int> vocab)
    {
        var pairFreq = new Dictionary<(string, string), int>();

        foreach (var (word, freq) in vocab)
        {
            var symbols = word.Split(' ');
            for (var i = 0; i < symbols.Length - 1; i++)
            {
                var pair = (symbols[i], symbols[i + 1]);
                if (!pairFreq.ContainsKey(pair))
                    pairFreq[pair] = 0;
                pairFreq[pair] += freq;
            }
        }

        return pairFreq.Select(p => new BPEPair(p.Key.Item1, p.Key.Item2, p.Value)).ToList();
    }

    /// <summary>
    ///     Merges the specified pair (left, right) into a new symbol.
    /// </summary>
    private Dictionary<string, int> MergePair(Dictionary<string, int> vocab, string left, string right,
        string newSymbol)
    {
        var newVocab = new Dictionary<string, int>();

        foreach (var (word, freq) in vocab)
        {
            var symbols = word.Split(' ').ToList();
            var newSymbols = new List<string>();

            var i = 0;
            while (i < symbols.Count)
                if (i < symbols.Count - 1 && symbols[i] == left && symbols[i + 1] == right)
                {
                    newSymbols.Add(newSymbol);
                    i += 2;
                }
                else
                {
                    newSymbols.Add(symbols[i]);
                    i++;
                }

            var joined = string.Join(" ", newSymbols);
            if (!newVocab.ContainsKey(joined))
                newVocab[joined] = 0;
            newVocab[joined] += freq;
        }

        return newVocab;
    }

    /// <summary>
    ///     Encodes text into a sequence of token IDs using learned BPE merges.
    /// </summary>
    public List<int> Encode(string text)
    {
        var tokens = new List<int>();

        foreach (var word in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
            var symbols = word.Select(c => c.ToString()).ToList();
            symbols.Add(Eow);

            foreach (var (left, right) in _merges)
            {
                var i = 0;
                while (i < symbols.Count - 1)
                    if (symbols[i] == left && symbols[i + 1] == right)
                    {
                        symbols[i] = left + right;
                        symbols.RemoveAt(i + 1);
                    }
                    else
                    {
                        i++;
                    }
            }

            foreach (var s in symbols)
                if (TokenToId.TryGetValue(s, out var id))
                    tokens.Add(id);
                else
                    tokens.Add(TokenToId["<UNK>"]);
        }

        return tokens;
    }

    /// <summary>
    ///     Decodes a list of token IDs back into text.
    /// </summary>
    public string Decode(List<int> tokenIds)
    {
        var words = tokenIds.Select(id => IdToToken.ContainsKey(id) ? IdToToken[id] : "<UNK>");
        var joined = string.Join("", words).Replace("</w>", " ");
        return joined.Trim();
    }

    /// <summary>
    ///     Creates training sequences of fixed length (context size) from a corpus.
    /// </summary>
    public List<List<int>> CreateSequences(IEnumerable<string> texts, int contextSize)
    {
        var allTokens = new List<int>();

        foreach (var text in texts)
            allTokens.AddRange(Encode(text));

        for (var i = 0; i <= allTokens.Count - contextSize; i += contextSize)
        {
            var seq = allTokens.Skip(i).Take(contextSize).ToList();
            Sequences.Add(seq);
        }

        Console.WriteLine($"Created {Sequences.Count} sequences of length {contextSize}");
        return Sequences;
    }
}