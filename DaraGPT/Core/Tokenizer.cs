using System.Collections.Concurrent;
using System.Runtime;
using System.Text;
using OpenCL.Net;
using ClProgram = OpenCL.Net.Program;
using Environment = System.Environment;

namespace DaraGPT;

public class Tokenizer : IDisposable
{
    private readonly object tokenLock = new();
    private bool gpuAvailable;
    private Context gpuContext;
    private Device gpuDevice;
    private ClProgram gpuProgram;
    private CommandQueue gpuQueue;
    private Kernel kernelCountPairs;
    private Dictionary<(int, int), int> mergeRank = new();

    private List<(int, int)> merges = new();

    private int nextId;

    public Tokenizer()
    {
        InitGPU();

        AddToken("<PAD>");
        AddToken("<UNK>");
        AddToken("<BOS>");
        AddToken("<EOS>");
    }

    public Dictionary<string, int> TokenToId { get; private set; } = new();
    public Dictionary<int, string> IdToToken { get; private set; } = new();


    public void Dispose()
    {
        if (gpuAvailable)
        {
            Cl.ReleaseKernel(kernelCountPairs);
            Cl.ReleaseProgram(gpuProgram);
            Cl.ReleaseCommandQueue(gpuQueue);
            Cl.ReleaseContext(gpuContext);
        }
    }

    private void InitGPU()
    {
        try
        {
            var platforms = Cl.GetPlatformIDs(out _);
            foreach (var platform in platforms)
            {
                var devices = Cl.GetDeviceIDs(platform, DeviceType.Gpu, out _);
                if (devices.Length > 0)
                {
                    gpuDevice = devices[0];
                    gpuContext = Cl.CreateContext(null, 1, new[] { gpuDevice }, null, IntPtr.Zero, out _);
                    gpuQueue = Cl.CreateCommandQueue(gpuContext, gpuDevice, CommandQueueProperties.None, out _);

                    var kernelSource = @"
                    __kernel void CountPairs(__global int* words, __global int* freqs, __global int* offsets, int totalWords, __global int2* pairs)
                    {
                        int gid = get_global_id(0);
                        if (gid >= totalWords) return;
                        int start = offsets[gid];
                        int end = offsets[gid + 1];
                        for (int i = start; i < end - 1; i++) {
                            pairs[i] = (int2)(words[i], words[i+1]);
                        }
                    }
                ";

                    gpuProgram = Cl.CreateProgramWithSource(gpuContext, 1, new[] { kernelSource }, null, out _);
                    var buildError = Cl.BuildProgram(gpuProgram, 1, new[] { gpuDevice }, string.Empty, null,
                        IntPtr.Zero);

                    if (buildError != ErrorCode.Success)
                    {
                        Console.WriteLine($"Neuspešna kompilacija OpenCL kernela ({buildError}).");
                        var buildLog = Cl.GetProgramBuildInfo(gpuProgram, gpuDevice, ProgramBuildInfo.Log, out _);
                        Console.WriteLine("=== OpenCL Build Log ===");
                        Console.WriteLine(buildLog.ToString());
                        Console.WriteLine("========================");
                        gpuAvailable = false;
                        return;
                    }

                    kernelCountPairs = Cl.CreateKernel(gpuProgram, "CountPairs", out var kernelErr);
                    if (kernelErr != ErrorCode.Success)
                    {
                        Console.WriteLine($"Greška pri kreiranju kernela: {kernelErr}");
                        gpuAvailable = false;
                        return;
                    }

                    var deviceName = Cl.GetDeviceInfo(gpuDevice, DeviceInfo.Name, out _).ToString();
                    Console.WriteLine($"GPU detektovan: {deviceName}");

                    // Isključi Intel GPU ako je prisutan
                    if (deviceName.Contains("Intel", StringComparison.OrdinalIgnoreCase))
                    {
                        Console.WriteLine(
                            "Intel GPU detektovan — preskačem GPU mod (poznat problem sa OpenCL kernelima).");
                        gpuAvailable = false;
                        return;
                    }

                    gpuAvailable = true;
                    break;
                }
            }

            if (!gpuAvailable)
                Console.WriteLine("Nijedan podržani GPU nije pronađen — prelazim na CPU mod.");
        }
        catch (Exception ex)
        {
            gpuAvailable = false;
            Console.WriteLine($"OpenCL GPU nije dostupan ({ex.Message}), prelazim na CPU mod.");
        }
    }

    public int AddToken(string token)
    {
        lock (tokenLock)
        {
            if (TokenToId.TryGetValue(token, out var id))
                return id;
            id = nextId++;
            TokenToId[token] = id;
            IdToToken[id] = token;
            return id;
        }
    }

    public void TrainBPE(IEnumerable<string> texts, int vocabTarget = 30000)
    {
        Console.WriteLine("Treniram BPE tokenizator...");

        GCSettings.LatencyMode = GCLatencyMode.SustainedLowLatency;
        var vocab = new ConcurrentDictionary<ulong, (List<int> seq, int freq)>();

        Parallel.ForEach(texts, text =>
        {
            foreach (var word in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
            {
                var chars = word.Select(c => AddToken(c.ToString())).ToList();
                chars.Add(AddToken("</w>"));

                var hash = 1469598103934665603UL;
                foreach (var t in chars)
                    hash = (hash ^ (ulong)t) * 1099511628211UL;

                vocab.AddOrUpdate(hash,
                    _ => (chars, 1),
                    (_, existing) => (existing.seq, existing.freq + 1));
            }
        });

        Console.WriteLine($"Inicijalni vokabular: {vocab.Count} reči");

        if (gpuAvailable)
            Console.WriteLine("Korisitm GPU za CountPairs algoritam.");

        while (TokenToId.Count < vocabTarget)
        {
            var pairFreq = gpuAvailable
                ? CountPairsGPU(vocab)
                : CountPairsCPU(vocab);

            if (pairFreq.Count == 0)
                break;

            var bestPair = pairFreq.Aggregate((a, b) => a.Value > b.Value ? a : b).Key;
            merges.Add(bestPair);

            var t1 = IdToToken.ContainsKey(bestPair.Item1)
                ? IdToToken[bestPair.Item1]
                : $"<{bestPair.Item1}>";
            var t2 = IdToToken.ContainsKey(bestPair.Item2)
                ? IdToToken[bestPair.Item2]
                : $"<{bestPair.Item2}>";
            AddToken(t1 + t2);

            var newVocab = new ConcurrentDictionary<ulong, (List<int> seq, int freq)>();
            Parallel.ForEach(vocab, kv =>
            {
                var word = kv.Value.seq;
                var freq = kv.Value.freq;
                var newWord = MergePair(word, bestPair);

                var hash = 1469598103934665603UL;
                foreach (var t in newWord)
                    hash = (hash ^ (ulong)t) * 1099511628211UL;

                newVocab.AddOrUpdate(hash,
                    _ => (newWord, freq),
                    (_, existing) => (existing.seq, existing.freq + freq));
            });

            vocab = newVocab;

            if (TokenToId.Count % 500 == 0)
                Console.WriteLine($"  • {TokenToId.Count} tokena");
        }

        mergeRank = new Dictionary<(int, int), int>(merges.Count);
        for (var i = 0; i < merges.Count; i++)
            mergeRank[merges[i]] = i;

        Console.WriteLine($"Završeno — ukupno {TokenToId.Count} tokena, {merges.Count} spajanja.");
    }

    private Dictionary<(int, int), int> CountPairsCPU(
        ConcurrentDictionary<ulong, (List<int> seq, int freq)> vocab)
    {
        // Lokalne mape po threadu (smanjuju lock contention)
        var global =
            new ConcurrentDictionary<(int, int), int>(Environment.ProcessorCount * 2, 1_000_000);

        Parallel.ForEach(vocab, () => new Dictionary<(int, int), int>(512),
            (kv, _, local) =>
            {
                var seq = kv.Value.seq;
                var freq = kv.Value.freq;

                for (var i = 0; i < seq.Count - 1; i++)
                {
                    var pair = (seq[i], seq[i + 1]);
                    if (local.TryGetValue(pair, out var old))
                        local[pair] = old + freq;
                    else
                        local[pair] = freq;
                }

                return local;
            },
            local =>
            {
                foreach (var kv in local)
                    global.AddOrUpdate(kv.Key, kv.Value, (_, old) => old + kv.Value);
            });

        return global.ToDictionary(x => x.Key, x => x.Value);
    }

    private Dictionary<(int, int), int> CountPairsGPU(
        ConcurrentDictionary<ulong, (List<int> seq, int freq)> vocab)
    {
        try
        {
            var sequences = vocab.Values.ToList();
            var allWords = sequences.SelectMany(s => s.seq).ToArray();
            var freqs = sequences.Select(s => s.freq).ToArray();
            var offsets = new int[sequences.Count + 1];
            var sum = 0;
            for (var i = 0; i < sequences.Count; i++)
            {
                offsets[i] = sum;
                sum += sequences[i].seq.Count;
            }

            offsets[sequences.Count] = sum;

            if (sum < 2)
                return new Dictionary<(int, int), int>();

            // GPU memorija
            var bufWords = Cl.CreateBuffer(gpuContext, MemFlags.CopyHostPtr | MemFlags.ReadOnly, allWords,
                out _);
            var bufFreqs = Cl.CreateBuffer(gpuContext, MemFlags.CopyHostPtr | MemFlags.ReadOnly, freqs, out _);
            var bufOffsets = Cl.CreateBuffer(gpuContext, MemFlags.CopyHostPtr | MemFlags.ReadOnly, offsets,
                out _);
            var bufPairs = Cl.CreateBuffer<int>(gpuContext, MemFlags.WriteOnly, sum * 2, out _);

            Cl.SetKernelArg(kernelCountPairs, 0, bufWords);
            Cl.SetKernelArg(kernelCountPairs, 1, bufFreqs);
            Cl.SetKernelArg(kernelCountPairs, 2, bufOffsets);
            Cl.SetKernelArg(kernelCountPairs, 3, sequences.Count);
            Cl.SetKernelArg(kernelCountPairs, 4, bufPairs);

            var globalWorkSize = new[] { new IntPtr(sequences.Count) };
            var err = Cl.EnqueueNDRangeKernel(gpuQueue, kernelCountPairs, 1, null, globalWorkSize, null, 0,
                null,
                out _);
            if (err != ErrorCode.Success)
                throw new Exception($"OpenCL kernel error: {err}");

            Cl.Finish(gpuQueue);

            var pairs = new int[sum * 2];
            Cl.EnqueueReadBuffer(gpuQueue, bufPairs, Bool.True, IntPtr.Zero,
                new IntPtr(pairs.Length * sizeof(int)),
                pairs, 0, null, out _);

            Cl.ReleaseMemObject(bufWords);
            Cl.ReleaseMemObject(bufFreqs);
            Cl.ReleaseMemObject(bufOffsets);
            Cl.ReleaseMemObject(bufPairs);

            // Ako GPU ne vrati ništa — pređi na CPU
            if (pairs.All(p => p == 0))
            {
                Console.WriteLine("GPU ne vraća podatke — prelazim na CPU mod.");
                gpuAvailable = false;
                return CountPairsCPU(vocab);
            }

            var result = new Dictionary<(int, int), int>(1024);
            for (var i = 0; i < pairs.Length - 1; i += 2)
            {
                var key = (pairs[i], pairs[i + 1]);
                if (!result.TryAdd(key, 1))
                    result[key]++;
            }

            return result;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"⚠️ GPU kernel neuspeo ({ex.Message}), prelazim na CPU mod.");
            gpuAvailable = false;
            return CountPairsCPU(vocab);
        }
    }


    private static List<int> MergePair(List<int> tokens, (int, int) pair)
    {
        var merged = new List<int>(tokens.Count);
        var i = 0;
        while (i < tokens.Count)
            if (i < tokens.Count - 1 && tokens[i] == pair.Item1 && tokens[i + 1] == pair.Item2)
            {
                merged.Add(tokens[i]);
                i += 2;
            }
            else
            {
                merged.Add(tokens[i]);
                i++;
            }

        return merged;
    }

    // Kodira pojedinačni tekst koristeći BPE spajanja
    private int[] EncodeSingle(string text)
    {
        var symbols = new List<int>();
        foreach (var c in text)
            symbols.Add(AddToken(c.ToString()));

        symbols.Add(AddToken("</w>"));

        // Spajanja po rangu
        while (true)
        {
            var pairRanks = new List<(int, int, int)>();
            for (var i = 0; i < symbols.Count - 1; i++)
            {
                var pair = (symbols[i], symbols[i + 1]);
                if (mergeRank.TryGetValue(pair, out var rank))
                    pairRanks.Add((i, pair.Item1, pair.Item2));
            }

            if (pairRanks.Count == 0)
                break;

            var best = pairRanks.OrderBy(p => mergeRank[(p.Item2, p.Item3)]).First();
            symbols = MergePair(symbols, (best.Item2, best.Item3));
        }

        return symbols.ToArray();
    }

    // Paralelno kodiranje više tekstova (brzo + thread-safe)
    public List<int[]> EncodeBatch(IEnumerable<string> texts)
    {
        if (mergeRank == null || mergeRank.Count == 0)
        {
            mergeRank = new Dictionary<(int, int), int>(merges.Count);
            for (var i = 0; i < merges.Count; i++)
                mergeRank[merges[i]] = i;
        }

        var results = new ConcurrentBag<int[]>();

        Parallel.ForEach(texts, text =>
        {
            try
            {
                var encoded = EncodeSingle(text);
                if (encoded.Length > 0)
                    results.Add(encoded);
            }
            catch
            {
                // preskoči greške u tokenizaciji pojedinačnih dokumenata
            }
        });

        return results.ToList();
    }


    public int[] Encode(string text)
    {
        return EncodeSingle(text);
    }


    public void Save(string path)
    {
        var dir = Path.GetDirectoryName(path);
        var nameNoExt = Path.GetFileNameWithoutExtension(path);
        if (string.IsNullOrWhiteSpace(nameNoExt)) nameNoExt = "tokenizer";
        if (string.IsNullOrEmpty(dir)) dir = ".";
        Directory.CreateDirectory(dir);

        var finalPath = Path.Combine(dir, nameNoExt + ".tokbin");

        using var fs = new FileStream(finalPath, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs, Encoding.UTF8, false);

        // Header
        bw.Write("TOKINT"); // magic identifikator
        bw.Write(1); // verzija formata

        // Token -> ID
        bw.Write(TokenToId.Count);
        foreach (var kv in TokenToId)
        {
            bw.Write(kv.Key);
            bw.Write(kv.Value);
        }

        // ID -> Token
        bw.Write(IdToToken.Count);
        foreach (var kv in IdToToken)
        {
            bw.Write(kv.Key);
            bw.Write(kv.Value ?? string.Empty);
        }

        // Merge parovi
        bw.Write(merges.Count);
        foreach (var (a, b) in merges)
        {
            bw.Write(a);
            bw.Write(b);
        }

        // Ostalo
        bw.Write(nextId);

        Console.WriteLine(
            $"Tokenizer sačuvan BIN ({TokenToId.Count} tokena, {merges.Count} spajanja) → {finalPath}");
    }

    public void Load(string path)
    {
        if (!File.Exists(path))
        {
            Console.WriteLine($"Tokenizer fajl nije pronađen: {path}");
            return;
        }

        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        using var br = new BinaryReader(fs, Encoding.UTF8, false);

        var magic = br.ReadString();
        if (magic != "TOKINT")
            throw new InvalidDataException("Nije ispravan TOK bin fajl.");

        var ver = br.ReadInt32();
        if (ver != 1)
            throw new NotSupportedException($"Nepodržana verzija tokenizatora: {ver}");

        // TokenToId
        var t2i = br.ReadInt32();
        TokenToId = new Dictionary<string, int>(t2i);
        for (var i = 0; i < t2i; i++)
        {
            var k = br.ReadString();
            var v = br.ReadInt32();
            TokenToId[k] = v;
        }

        // IdToToken
        var i2t = br.ReadInt32();
        IdToToken = new Dictionary<int, string>(i2t);
        for (var i = 0; i < i2t; i++)
        {
            var k = br.ReadInt32();
            var v = br.ReadString();
            IdToToken[k] = v;
        }

        // Merge parovi
        var mc = br.ReadInt32();
        merges = new List<(int, int)>(mc);
        for (var i = 0; i < mc; i++)
        {
            var a = br.ReadInt32();
            var b = br.ReadInt32();
            merges.Add((a, b));
        }

        // Sledeći ID
        nextId = br.ReadInt32();

        // Rekonstruiši rangove spajanja
        mergeRank = new Dictionary<(int, int), int>(merges.Count);
        for (var i = 0; i < merges.Count; i++)
            mergeRank[merges[i]] = i;

        Console.WriteLine($"Tokenizer učitan BIN ({TokenToId.Count} tokena, {merges.Count} spajanja) ← {path}");
    }
}