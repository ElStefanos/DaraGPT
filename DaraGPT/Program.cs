using System;
using System.IO;
using Newtonsoft.Json;
using System.Threading.Tasks;

namespace DaraGPT
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            Console.WriteLine("=== DaraGPT ===");

            string configPath = "./config.json";
            Config cfg;

            if (File.Exists(configPath))
            {
                try
                {
                    string json = File.ReadAllText(configPath);
                    cfg = JsonConvert.DeserializeObject<Config>(json)!;
                    Console.WriteLine("Učitan konfiguracioni fajl: config.json");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Greška pri učitavanju config.json ({ex.Message}). Koristim podrazumevane vrednosti.");
                    cfg = GetDefaultConfig();
                }
            }
            else
            {
                Console.WriteLine("Nema config.json fajla, kreiram novi sa podrazumevanim vrednostima...");
                cfg = GetDefaultConfig();
                string json = JsonConvert.SerializeObject(cfg, Formatting.Indented);
                File.WriteAllText(configPath, json);
                Console.WriteLine("Kreiran novi config.json.");
            }

            Console.WriteLine("Izaberi opciju:");
            Console.WriteLine("1. Trening (lokalni fajlovi)");
            Console.WriteLine("2. Online trening (Wikipedia → korpus → trening)");
            Console.WriteLine("3. Razgovor sa modelom");
            Console.Write("> ");
            var izbor = Console.ReadLine();

            var tokenizerPath = Path.Combine(cfg.CheckpointDir, "tokenizer.tokbin");
            var modelPath = Path.Combine(cfg.CheckpointDir, "model.bin");

            switch (izbor)
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

        static Config GetDefaultConfig() => new Config
        {
            DModel = 256,
            NumLayers = 6,
            ContextSize = 500,
            VocabSize = 30000,
            LearningRate = 1e-3f,
            DevicePreference = "AMD",
            CheckpointDir = "checkpoints"
        };

        static void RunTraining(Config cfg, string tokenizerPath, string modelPath)
        {
            Console.WriteLine("\nPokrećem trening...");

            var tokenizer = new Tokenizer();
            if (!File.Exists(tokenizerPath))
            {
                var allTexts = Directory.GetFiles("Data", "*.txt", SearchOption.AllDirectories)
                    .Select(File.ReadAllText);
                tokenizer.TrainBPE(allTexts, vocabTarget: cfg.VocabSize);
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
            if (!int.TryParse(Console.ReadLine(), out int epochs)) epochs = 3;

            trainer.Train(epochs);

            Directory.CreateDirectory(cfg.CheckpointDir);
            tokenizer.Save(tokenizerPath);
            model.Save(cfg.CheckpointDir);

            Console.WriteLine("Trening završen i model sačuvan.");
        }

        static async Task RunWikipediaTraining(Config cfg, string tokenizerPath, string modelPath)
        {
            Console.WriteLine("\nPokrećem preuzimanje Wikipedia stranica...");

            Console.Write("Koliko stranica da preuzmem: ");
            if (!int.TryParse(Console.ReadLine(), out int maxPages))
                maxPages = 5;

            var wiki = new WikipediaTrainer(cfg);
            await wiki.DownloadWikipediaAsync(maxPages);

            Console.WriteLine("\nPreuzimanje završeno. Pokrećem trening na novim podacima...");
            RunTraining(cfg, tokenizerPath, modelPath);
        }

        static void RunChat(Config cfg, string tokenizerPath, string modelPath)
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

            int V = Math.Max(tokenizer.TokenToId.Count, 4);
            Console.WriteLine("Model spreman! (unesi 'exit' za izlaz)\n");

            while (true)
            {
                Console.Write("Ti: ");
                var input = Console.ReadLine();
                if (input == null || input.Trim().ToLower() == "exit") break;

                var tokens = tokenizer.Encode(input);
                var generated = new List<int>(tokens);

                int maxSteps = 80;
                int maxContext = cfg.ContextSize;
                string answer = "";

                Console.Write("DaraGPT: ");

                for (int step = 0; step < maxSteps; step++)
                {
                    if (generated.Count > maxContext)
                        generated = generated.Skip(generated.Count - maxContext).ToList();

                    var logits = model.ForwardTokens(generated.ToArray());
                    int lastRowStart = Math.Max(0, (generated.Count - 1) * V);

                    int bestId = 0;
                    float maxVal = float.NegativeInfinity;

                    var freq = new Dictionary<int, int>();
                    foreach (var t in generated)
                        freq[t] = freq.ContainsKey(t) ? freq[t] + 1 : 1;

                    for (int i = 0; i < V && lastRowStart + i < logits.Length; i++)
                    {
                        float v = logits[lastRowStart + i];
                        if (freq.TryGetValue(i, out int count) && count > 1)
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
                    System.Threading.Thread.Sleep(25);
                }

                Console.WriteLine("\n");
            }
        }
    }
}
