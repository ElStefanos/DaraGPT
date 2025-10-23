using System.Diagnostics;
using OpenCL.Net;

namespace DaraGPT;

public class Trainer
{
    private readonly TextDataset dataset;
    private readonly GPTModel model;
    private readonly Tokenizer tokenizer;

    public Trainer(GPTModel model, Tokenizer tokenizer, TextDataset dataset)
    {
        this.model = model;
        this.tokenizer = tokenizer;
        this.dataset = dataset;
    }

    public void Train(int epochs = 3)
    {
        Console.WriteLine("Trening u toku...");

        var gpu = model != null ? model.cfg != null ? new GpuEngine(model.cfg.DevicePreference) : model.gpu : null;

        if (gpu != null && gpu.Available)
            Console.WriteLine($"GPU aktivan: {Cl.GetDeviceInfo(gpu.Device, DeviceInfo.Name, out _)}");
        else
            Console.WriteLine("GPU nije dostupan, koristi se CPU fallback.");

        for (var e = 0; e < epochs; e++)
        {
            var batches = 0;
            double epochLoss = 0;

            // Keširaj batch-eve jednom
            var batchSize = 8; // možeš menjati
            var batchesList = dataset.GetBatches(model.cfg.ContextSize, batchSize).ToList();
            var totalBatches = batchesList.Count;

            Console.WriteLine($"\n[Epoch {e + 1}] Dataset spreman. Ukupno batch-eva: {totalBatches}");

            var epochTimer = Stopwatch.StartNew();

            foreach (var (inputs, targets) in batchesList)
            {
                batches++;
                double batchLoss = 0;

                for (var b = 0; b < inputs.Length; b++)
                {
                    var input = inputs[b];
                    var target = targets[b];
                    var T = input.Length;
                    var V = model.cfg.VocabSize;

                    var hidden = model.ForwardHidden(input);
                    var logits = model.ProjectToVocab(hidden, T);
                    var probs = LinAlgCPU.SoftmaxRowwise(logits, T, V);

                    var gradLogits = new float[probs.Length];
                    for (var t = 0; t < T; t++)
                    {
                        var targetId = target[t];
                        var row = t * V;
                        var pTrue = Math.Max(1e-12, probs[row + targetId]);
                        batchLoss += -Math.Log(pTrue);

                        for (var i = 0; i < V; i++) gradLogits[row + i] = probs[row + i];
                        gradLogits[row + targetId] -= 1f;
                    }

                    model.BackwardOnOutput(hidden, gradLogits, T, model.cfg.LearningRate);
                }

                epochLoss += batchLoss / Math.Max(1, inputs.Length);
                Console.WriteLine($"Epoch {e + 1} | Batch {batches}/{totalBatches} | Loss {epochLoss / batches:F4}");
            }

            epochTimer.Stop();
            Console.WriteLine(
                $"Epoch {e + 1}/{epochs} završena. AvgLoss={epochLoss / Math.Max(1, batches):F4} | Trajanje: {epochTimer.Elapsed.TotalSeconds:F1}s");
        }

        Console.WriteLine("\nTrening završen.");
    }
}