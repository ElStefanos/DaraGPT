using System.Collections.Concurrent;

namespace DaraGPT;

public class TextDataset
{
    private readonly string? dataDir;
    private readonly Tokenizer tokenizer;

    public TextDataset(string dataDir, Tokenizer tokenizer)
    {
        this.dataDir = dataDir;
        this.tokenizer = tokenizer;
    }

    public List<int[]> Sequences { get; } = new();

    public void LoadAllTextFiles()
    {
        if (dataDir == null)
            throw new InvalidOperationException(
                "Ovaj dataset je kreiran bez 'Data' direktorijuma i ne može učitavati fajlove.");

        if (!Directory.Exists(dataDir))
            throw new DirectoryNotFoundException(dataDir);

        var files = Directory.GetFiles(dataDir, "*.txt", SearchOption.AllDirectories);
        Console.WriteLine($"Pronađeno {files.Length} fajlova. Učitavam i tokenizujem paralelno...");

        var texts = new ConcurrentBag<string>();

        // Brzo učitavanje svih tekstova paralelno
        Parallel.ForEach(files, f =>
        {
            try
            {
                var text = File.ReadAllText(f);
                if (!string.IsNullOrWhiteSpace(text))
                    texts.Add(text);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Greška pri čitanju {f}: {ex.Message}");
            }
        });

        Console.WriteLine(
            $"Učitano {texts.Count} tekstova. Pokrećem tokenizaciju ({Environment.ProcessorCount} niti + GPU fallback)...");

        // Tokenizacija u delovima (batch od 50 fajlova radi stabilnosti memorije)
        var chunkSize = 50;
        var textList = texts.ToList();

        Parallel.ForEach(
            Partitioner.Create(0, textList.Count, chunkSize),
            range =>
            {
                var localSeq = new List<int[]>();
                var slice = textList.Skip(range.Item1).Take(range.Item2 - range.Item1);
                var encoded = tokenizer.EncodeBatch(slice);

                foreach (var tokens in encoded)
                    if (tokens.Length > 1)
                        localSeq.Add(tokens);

                lock (Sequences)
                {
                    Sequences.AddRange(localSeq);
                }
            });

        Console.WriteLine($"Tokenizacija završena — ukupno {Sequences.Count} sekvenci ({texts.Count} fajlova).");
    }

    public void AddDataFromTokens(int[] tokens)
    {
        if (tokens != null && tokens.Length > 1)
            Sequences.Add(tokens);
    }

    public void AddDataFromTokens(int[] tokens, int contextSize, bool preSlice)
    {
        if (!preSlice)
        {
            AddDataFromTokens(tokens);
            return;
        }

        if (tokens == null || tokens.Length <= 1)
            return;

        for (var i = 0; i < tokens.Length - 1; i += contextSize)
        {
            var len = Math.Min(contextSize + 1, tokens.Length - i);
            var slice = new int[len];
            Array.Copy(tokens, i, slice, 0, len);
            Sequences.Add(slice);
        }
    }

    public IEnumerable<(int[][] inputs, int[][] targets)> GetBatches(int contextSize, int batchSize)
    {
        var all = new List<int>();
        foreach (var seq in Sequences)
        {
            all.AddRange(seq);
            all.Add(0); // separator token
        }

        var total = (all.Count - 1) / contextSize;
        for (var i = 0; i < total; i += batchSize)
        {
            var batchInputs = new List<int[]>();
            var batchTargets = new List<int[]>();

            for (var b = 0; b < batchSize && i + b < total; b++)
            {
                var start = (i + b) * contextSize;
                var len = Math.Min(contextSize, all.Count - 1 - start);

                var input = new int[len];
                var target = new int[len];
                all.CopyTo(start, input, 0, len);
                all.CopyTo(start + 1, target, 0, len);

                batchInputs.Add(input);
                batchTargets.Add(target);
            }

            yield return (batchInputs.ToArray(), batchTargets.ToArray());
        }
    }
}