using System.Runtime;
using System.Text;
using Newtonsoft.Json;

namespace DaraGPT;

internal class Program
{
    private static async Task Main(string[] args)
    {
        GCSettings.LatencyMode = GCLatencyMode.SustainedLowLatency;
        ThreadPool.SetMinThreads(Environment.ProcessorCount, Environment.ProcessorCount);

        Console.OutputEncoding = Encoding.UTF8;
        Console.WriteLine("=== DaraGPT ===");

        var configPath = "./config.json";
        Config cfg;

        if (File.Exists(configPath))
        {
            try
            {
                var json = File.ReadAllText(configPath);
                cfg = JsonConvert.DeserializeObject<Config>(json)!;
                Console.WriteLine("Učitan konfiguracioni fajl: config.json");
            }
            catch (Exception ex)
            {
                Console.WriteLine(
                    $"Greška pri učitavanju config.json ({ex.Message}). Koristim podrazumevane vrednosti.");
                cfg = GetDefaultConfig();
            }
        }
        else
        {
            Console.WriteLine("Nema config.json fajla, kreiram novi sa podrazumevanim vrednostima...");
            cfg = GetDefaultConfig();
            var json = JsonConvert.SerializeObject(cfg, Formatting.Indented);
            File.WriteAllText(configPath, json);
            Console.WriteLine("Kreiran novi config.json.");
        }

        Console.WriteLine("Izaberi opciju:");
        Console.WriteLine("1. Trening (lokalni fajlovi)");
        Console.WriteLine("2. Online trening (Wikipedia → korpus → trening)");
        Console.WriteLine("3. Razgovor sa modelom");
        Console.Write("> ");
        var option = Console.ReadLine();

        var tokenizerPath = Path.Combine(cfg.CheckpointDir, "tokenizer.tokbin");
        var modelPath = Path.Combine(cfg.CheckpointDir, "model.bin");

        switch (option)
        {
            case "1":
                RunTraining(cfg, tokenizerPath, modelPath);
                break;

            case "2":
                await RunWikipediaTraining(cfg, tokenizerPath, modelPath);
                break;

            case "3":
                RunChat(cfg, tokenizerPath, modelPath);
                break;

            default:
                Console.WriteLine("Nepoznata opcija.");
                break;
        }
    }

    private static Config GetDefaultConfig()
    {
        return new Config
        {
            DModel = 256,
            NumLayers = 6,
            ContextSize = 500,
            VocabSize = 30000,
            LearningRate = 1e-3f,
            DevicePreference = "AMD",
            CheckpointDir = "checkpoints"
        };
    }

    private static void RunTraining(Config cfg, string tokenizerPath, string modelPath)
    {
        Console.WriteLine("\nPokrećem trening...");

        var tokenizer = new Tokenizer();
        if (!File.Exists(tokenizerPath))
        {
            var allTexts = Directory.GetFiles("Data", "*.txt", SearchOption.AllDirectories)
                .Select(File.ReadAllText);
            tokenizer.TrainBPE(allTexts, cfg.VocabSize);
            tokenizer.Save(tokenizerPath);
        }
        else
        {
            tokenizer.Load(tokenizerPath);
        }


        var dataset = new TextDataset("Data", tokenizer);
        dataset.LoadAllTextFiles();

        cfg.VocabSize = Math.Max(tokenizer.TokenToId.Count, 4);

        var model = new GPTModel(cfg);
        var trainer = new Trainer(model, tokenizer, dataset);

        Console.Write("Unesi broj epoha: ");
        if (!int.TryParse(Console.ReadLine(), out var epochs)) epochs = 3;

        trainer.Train(epochs);

        Directory.CreateDirectory(cfg.CheckpointDir);
        tokenizer.Save(tokenizerPath);
        model.Save(cfg.CheckpointDir);

        Console.WriteLine("Trening završen i model sačuvan.");
    }

    private static async Task RunWikipediaTraining(Config cfg, string tokenizerPath, string modelPath)
    {
        Console.WriteLine("\nPokrećem preuzimanje Wikipedia stranica...");

        Console.Write("Koliko stranica da preuzmem: ");
        if (!int.TryParse(Console.ReadLine(), out var maxPages))
            maxPages = 5;

        var wiki = new WikipediaTrainer(cfg);
        await wiki.DownloadWikipediaAsync(maxPages);

        Console.WriteLine("\nPreuzimanje završeno. Pokrećem trening na novim podacima...");
        RunTraining(cfg, tokenizerPath, modelPath);
    }

    private static void RunChat(Config cfg, string tokenizerPath, string modelPath)
    {
        if (!File.Exists(modelPath))
        {
            Console.WriteLine("Model nije pronađen. Pokreni trening prvo.");
            return;
        }

        Console.WriteLine("\nUčitavam model...");
        var tokenizer = new Tokenizer();
        tokenizer.Load(tokenizerPath);
        var model = GPTModel.Load(modelPath);

        var V = Math.Max(tokenizer.TokenToId.Count, 4);
        Console.WriteLine("Model spreman! (unesi 'exit' za izlaz)\n");

        while (true)
        {
            Console.Write("Ti: ");
            var input = Console.ReadLine();
            if (input == null || input.Trim().ToLower() == "exit") break;

            var tokens = tokenizer.Encode(input);
            var generated = new List<int>(tokens);

            var maxSteps = 80;
            var maxContext = cfg.ContextSize;
            var answer = "";

            Console.Write("DaraGPT: ");

            for (var step = 0; step < maxSteps; step++)
            {
                if (generated.Count > maxContext)
                    generated = generated.Skip(generated.Count - maxContext).ToList();

                var logits = model.ForwardTokens(generated.ToArray());
                var lastRowStart = Math.Max(0, (generated.Count - 1) * V);

                var bestId = 0;
                var maxVal = float.NegativeInfinity;

                var freq = new Dictionary<int, int>();
                foreach (var t in generated)
                    freq[t] = freq.ContainsKey(t) ? freq[t] + 1 : 1;

                for (var i = 0; i < V && lastRowStart + i < logits.Length; i++)
                {
                    var v = logits[lastRowStart + i];
                    if (freq.TryGetValue(i, out var count) && count > 1)
                        v -= 0.05f * count;

                    if (v > maxVal)
                    {
                        maxVal = v;
                        bestId = i;
                    }
                }

                var token = tokenizer.IdToToken.ContainsKey(bestId)
                    ? tokenizer.IdToToken[bestId]
                    : "<UNK>";

                if (token == "<PAD>" || token == "<EOS>")
                    break;

                generated.Add(bestId);
                answer += token + " ";

                Console.Write(token.Replace("</w>", "") + " ");
                Console.Out.Flush();
                Thread.Sleep(25);
            }

            Console.WriteLine("\n");
        }
    }
}