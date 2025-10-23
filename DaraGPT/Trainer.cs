using System;
using OpenCL.Net;

namespace DaraGPT
{
    public class Trainer
    {
        private readonly GPTModel model;
        private readonly Tokenizer tokenizer;
        private readonly TextDataset dataset;

        public Trainer(GPTModel model, Tokenizer tokenizer, TextDataset dataset)
        {
            this.model = model;
            this.tokenizer = tokenizer;
            this.dataset = dataset;
        }

        public void Train(int epochs = 3)
        {
            Console.WriteLine("Trening u toku...");
            
            var gpu = model != null ? (model.cfg != null ? new GpuEngine(model.cfg.DevicePreference) : model.gpu) : null;

            if (gpu != null && gpu.Available)
                Console.WriteLine($"GPU aktivan: {Cl.GetDeviceInfo(gpu.Device, DeviceInfo.Name, out _)}");
            else
                Console.WriteLine("GPU nije dostupan, koristi se CPU fallback.");

            for (int e = 0; e < epochs; e++)
            {
                int batches = 0;
                double epochLoss = 0;

                // Keširaj batch-eve jednom
                var batchesList = dataset.GetBatches(model.cfg.ContextSize).ToList();
                int totalBatches = batchesList.Count;

                Console.WriteLine($"\n[Epoch {e + 1}] Dataset spreman. Ukupno batch-eva: {totalBatches}");

                var epochTimer = System.Diagnostics.Stopwatch.StartNew();

                foreach (var (input, target) in batchesList)
                {
                    batches++;
                    int T = input.Length;
                    int V = model.cfg.VocabSize;

                    var hidden = model.ForwardHidden(input);        // (T x D)
                    var logits = model.ProjectToVocab(hidden, T);   // (T x V)

                    // Softmax
                    float[] probs = (gpu != null && gpu.Available)
                        ? gpu.RowSoftmax(logits, T, V)
                        : LinAlgCPU.SoftmaxRowwise(logits, T, V);

                    // CE loss + gradLogits
                    var gradLogits = new float[probs.Length];
                    double batchLoss = 0.0;

                    for (int t = 0; t < T; t++)
                    {
                        int targetId = target[t];
                        int row = t * V;
                        
                        if (targetId < 0 || targetId >= V)
                        {
                            // Ako token ne postoji u vokabularu, preskoči batch
                            continue;
                        }

                        double pTrue = Math.Max(1e-12, probs[row + targetId]);
                        batchLoss += -Math.Log(pTrue);

                        for (int i = 0; i < V; i++) gradLogits[row + i] = probs[row + i];
                        gradLogits[row + targetId] -= 1f;
                    }

                    batchLoss /= Math.Max(1, T);
                    epochLoss += batchLoss;

                    // Backward: izlazni sloj
                    model.BackwardOnOutput(hidden, gradLogits, T, model.cfg.LearningRate);

                    float gradSum = 0;
                    for (int i = 0; i < gradLogits.Length; i++) gradSum += Math.Abs(gradLogits[i]);
                    if (batches % 10 == 0) Console.WriteLine($"Grad sum: {gradSum}");
                    
                    // Backward: slojevi (minimalni demo backprop)
                    for (int i = model.Layers.Count - 1; i >= 0; i--)
                    {
                        hidden = model.Layers[i].Backward(hidden, hidden, T, model.cfg.LearningRate);
                    }

                    Console.WriteLine($"Epoch {e + 1} | Batch {batches}/{totalBatches} | Loss {epochLoss / batches:F4}");
                }

                epochTimer.Stop();
                Console.WriteLine($"Epoch {e + 1}/{epochs} završena. AvgLoss={epochLoss / Math.Max(1, batches):F4} | Trajanje: {epochTimer.Elapsed.TotalSeconds:F1}s");
            }

            Console.WriteLine("\nTrening završen.");
        }
    }
}
