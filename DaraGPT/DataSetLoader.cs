using System;
using System.Collections.Generic;
using System.IO;
using System.Linq; // <-- potrebno za Skip/Take

namespace DaraGPT
{
    public class TextDataset
    {
        private readonly string? dataDir;
        private readonly Tokenizer tokenizer;
        public List<int[]> Sequences { get; set; } = new List<int[]>();

        // Za lokalne fajlove
        public TextDataset(string dataDir, Tokenizer tokenizer)
        {
            this.dataDir = dataDir;
            this.tokenizer = tokenizer;
        }

        public void LoadAllTextFiles()
        {
            if (dataDir == null)
                throw new InvalidOperationException("Ovaj dataset je kreiran bez 'dataDir' i ne može učitavati fajlove.");

            if (!Directory.Exists(dataDir))
                throw new DirectoryNotFoundException(dataDir);

            var files = Directory.GetFiles(dataDir, "*.txt", SearchOption.AllDirectories);
            foreach (var f in files)
            {
                var text = File.ReadAllText(f);
                var tokens = tokenizer.Encode(text);
                if (tokens.Length > 1) Sequences.Add(tokens);
            }
        }

        // Minimalno: samo dodaj celu sekvencu tokena; seckanje radi GetBatches
        public void AddDataFromTokens(int[] tokens)
        {
            if (tokens != null && tokens.Length > 1)
                Sequences.Add(tokens);
        }

        // Ako želiš unapred da isečeš, možeš opcioni parametar:
        public void AddDataFromTokens(int[] tokens, int contextSize, bool preSlice)
        {
            if (!preSlice)
            {
                AddDataFromTokens(tokens);
                return;
            }

            if (tokens == null || tokens.Length <= 1) return;

            for (int i = 0; i < tokens.Length - 1; i += contextSize)
            {
                var len = Math.Min(contextSize + 1, tokens.Length - i); // +1 zbog target pomeraja
                var slice = new int[len];
                Array.Copy(tokens, i, slice, 0, len);
                Sequences.Add(slice);
            }
        }

        public IEnumerable<(int[], int[])> GetBatches(int contextSize)
        {
            // Spoji sve sekvence u jedan niz
            var all = new List<int>();
            foreach (var seq in Sequences)
            {
                all.AddRange(seq);
                all.Add(0); // opciono dodaj <EOS> (ako 0 = <PAD> / <EOS>)
            }

            // 2Pravi batcheve globalno
            for (int i = 0; i < all.Count - 1; i += contextSize)
            {
                int len = Math.Min(contextSize, all.Count - 1 - i);
                var input  = new int[len];
                var target = new int[len];
                all.CopyTo(i,     input,  0, len);
                all.CopyTo(i + 1, target, 0, len);
                yield return (input, target);
            }
        }
    }
}
