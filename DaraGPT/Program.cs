using DaraGPT.Math.Gpu;
using DaraGPT.Model;
using DaraGPT.Train;

namespace DaraGPT;

internal class Program
{
    private static void Main(string[] args)
    {

        var config = Config.Config.LoadOrCreate();
        var naming = config.ModelName;

        if (args.Length > 1 && args[0] == "-naming")
            naming = config.ModelName + $"_{args[1]}";
        
        var file = Path.Combine(config.DataPath, "Wikipedija", "Antarktik.txt");
        if (args.Length > 2 && args[2] == "-file" && args.Length > 3)
            file = Path.Combine(config.DataPath, args[3]);

        if (!File.Exists(file))
        {
            Console.WriteLine($"[Error] Dataset file not found: {file}");
            return;
        }

        Console.WriteLine($"[Tokenizer] Processing file: {file}");


        Tokenizer.Tokenizer tok = new(naming);
        tok.RunTokenizer(file, config.VocabSize, config.ContextSize);
        Console.WriteLine($"[Tokenizer] Created {tok.BTokenizer.Sequences.Count} sequences."); // 


        var gpu = new GpuEngine(config.DevicePreference);
        if (gpu.Available)
        {
            gpu.LoadKernel();
            gpu.CompileKernel();
        }
        else
        {
            Console.WriteLine("[GPU] Not available, continuing on CPU.");
        }


        var model = new GPTModel(gpu, config);
        Console.WriteLine($"[Model] Initialized {config.ModelName} ({config.DModel}d, {config.Head} heads).");


        var T = config.ContextSize;
        var padId = 0;
        var sequences = new List<int[]>(tok.BTokenizer.Sequences.Count);
        foreach (var s in tok.BTokenizer.Sequences)
            sequences.Add(PadOrTruncate(s, T, padId));

        if (sequences.Count == 0)
        {
            Console.WriteLine("[Error] No sequences to train on.");
            return;
        }
        
        var trainer = new Trainer(model, config.LearningRate, gpu);
        var epochs = 1; 
        var batchSize = 8; 
        var lossOnlyLast = true;

        Console.WriteLine(
            $"[Train] Starting training: epochs={epochs}, batchSize={batchSize}, T={T}, LR={config.LearningRate}");

        var rnd = new Random(1234);

        for (var ep = 1; ep <= epochs; ep++)
        {
            sequences = sequences.OrderBy(_ => rnd.Next()).ToList();

            var N = sequences.Count;
            var batches = (N + batchSize - 1) / batchSize;
            var avgLoss = 0.0;

            var idx = 0;
            for (var b = 1; b <= batches; b++)
            {
                var effectiveBatchSize = batchSize;
                if (idx + effectiveBatchSize > N)
                    effectiveBatchSize = N - idx;

                var batch = new List<int[]>(effectiveBatchSize);

                for (var i = 0; i < batchSize && idx < N; i++, idx++)
                    batch.Add(sequences[idx]);

                var loss = trainer.TrainBatch(batch, lossOnlyLast);
                avgLoss += loss;

                if (b % 10 == 0)
                    Console.WriteLine($"[Epoch {ep}] Batch {b}/{batches} | Loss {loss:0.0000}");
            }

            avgLoss /= MathF.Max(1, batches);
            Console.WriteLine($"[Epoch {ep}] AvgLoss = {avgLoss:0.0000}");
            
            var ckptPath = Path.Combine(config.ModelSavePath, $"{naming}_ep{ep}.modbin");
            Directory.CreateDirectory(config.ModelSavePath);
            SaveLoad.SaveModel(model, ckptPath); // 
            Console.WriteLine($"[Checkpoint] Saved to {ckptPath}");
        }


        var savePath = Path.Combine(config.ModelSavePath, naming + ".modbin");
        try
        {
            Directory.CreateDirectory(config.ModelSavePath);
            SaveLoad.SaveModel(model, savePath); // 
            Console.WriteLine($"[Model] Saved to {savePath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Error] Failed to save model: {ex.Message}");
        }

        Console.WriteLine("[Done] Training complete.");
    }
    
    private static int[] PadOrTruncate(IEnumerable<int> src, int T, int padId)
    {
        var arr = src as int[] ?? src.ToArray();
        if (arr.Length == T) return arr;
        if (arr.Length > T) return arr.Take(T).ToArray();

        var outArr = new int[T];
        Array.Copy(arr, outArr, arr.Length);
        for (var i = arr.Length; i < T; i++) outArr[i] = padId;
        return outArr;
    }
}