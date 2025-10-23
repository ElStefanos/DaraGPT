using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Newtonsoft.Json;

namespace DaraGPT
{
    public class Tokenizer
    {
        public Dictionary<string, int> TokenToId { get; private set; } = new();
        public Dictionary<int, string> IdToToken { get; private set; } = new();
        private int nextId = 0;

        // za BPE
        private Dictionary<(string, string), int> pairFreq = new();
        private List<(string, string)> merges = new();

        public Tokenizer()
        {
            AddToken("<PAD>");
            AddToken("<UNK>");
            AddToken("<BOS>");
            AddToken("<EOS>");
        }

        public int AddToken(string token)
        {
            if (TokenToId.TryGetValue(token, out var id)) return id;
            id = nextId++;
            TokenToId[token] = id;
            IdToToken[id] = token;
            return id;
        }

        // trenira BPE vokabular iz korpusa (npr. Wikipedia)
        public void TrainBPE(IEnumerable<string> texts, int vocabTarget = 150000)
        {
            Console.WriteLine("Treniram BPE tokenizator...");

            // razdvoji sve tekstove u liste karaktera
            var words = new List<List<string>>();
            foreach (var text in texts)
            {
                foreach (var word in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
                {
                    var chars = word.Select(c => c.ToString()).ToList();
                    chars.Add("</w>");
                    words.Add(chars);
                }
            }

            // inicijalni vokabular
            var vocab = new Dictionary<string, int>();
            foreach (var w in words)
            {
                string joined = string.Join(" ", w);
                if (!vocab.ContainsKey(joined)) vocab[joined] = 0;
                vocab[joined]++;
            }

            // dok ne dostignemo veličinu vokabulara
            while (TokenToId.Count < vocabTarget)
            {
                pairFreq.Clear();

                foreach (var (word, freq) in vocab)
                {
                    var symbols = word.Split(' ');
                    for (int i = 0; i < symbols.Length - 1; i++)
                    {
                        var pair = (symbols[i], symbols[i + 1]);
                        if (!pairFreq.ContainsKey(pair)) pairFreq[pair] = 0;
                        pairFreq[pair] += freq;
                    }
                }

                if (pairFreq.Count == 0) break;

                var bestPair = pairFreq.OrderByDescending(p => p.Value).First().Key;
                merges.Add(bestPair);

                // ažuriraj vokabular
                var newVocab = new Dictionary<string, int>();
                foreach (var (word, freq) in vocab)
                {
                    var pattern = $"{bestPair.Item1} {bestPair.Item2}";
                    var replacement = $"{bestPair.Item1}{bestPair.Item2}";
                    var newWord = word.Replace(pattern, replacement);
                    if (!newVocab.ContainsKey(newWord)) newVocab[newWord] = 0;
                    newVocab[newWord] += freq;
                }

                vocab = newVocab;
                AddToken(bestPair.Item1 + bestPair.Item2);

                if (TokenToId.Count % 500 == 0)
                    Console.WriteLine($"  • BPE merge {TokenToId.Count} tokena");
            }

            Console.WriteLine($"BPE trening završen. Ukupno {TokenToId.Count} tokena.");
        }

        // koristi BPE spajanja za kodiranje
        public int[] Encode(string text)
        {
            var ids = new List<int>();
            foreach (var word in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
            {
                var chars = word.Select(c => c.ToString()).ToList();
                chars.Add("</w>");

                foreach (var (a, b) in merges)
                {
                    for (int i = 0; i < chars.Count - 1; i++)
                    {
                        if (chars[i] == a && chars[i + 1] == b)
                        {
                            chars[i] = a + b;
                            chars.RemoveAt(i + 1);
                            i--;
                        }
                    }
                }

                foreach (var token in chars)
                {
                    if (TokenToId.TryGetValue(token, out int id))
                        ids.Add(id);
                    else
                        ids.Add(TokenToId["<UNK>"]);
                }
            }

            return ids.ToArray();
        }

        public string Decode(int[] ids)
        {
            var sb = new StringBuilder();
            foreach (var id in ids)
            {
                if (IdToToken.TryGetValue(id, out var t))
                    sb.Append(t.Replace("</w>", "")).Append(' ');
                else
                    sb.Append("<UNK> ");
            }

            return sb.ToString().TrimEnd();
        }

        // sačuvaj i merges
        public void Save(string path)
        {
            // Forsiraj .tokbin ekstenziju i smislen izlazni put
            var dir = Path.GetDirectoryName(path);
            var nameNoExt = Path.GetFileNameWithoutExtension(path);
            if (string.IsNullOrWhiteSpace(nameNoExt)) nameNoExt = "tokenizer";

            if (string.IsNullOrEmpty(dir)) dir = ".";
            Directory.CreateDirectory(dir);

            var finalPath = Path.Combine(dir, nameNoExt + ".tokbin");

            using var fs = new FileStream(finalPath, FileMode.Create, FileAccess.Write, FileShare.None);
            using var bw = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: false);

            bw.Write("TOK");    // magic
            bw.Write(1);        // version

            // TokenToId
            bw.Write(TokenToId.Count);
            foreach (var kv in TokenToId)
            {
                bw.Write(kv.Key);
                bw.Write(kv.Value);
            }

            // IdToToken (redundantno ali zgodno za brz decode)
            bw.Write(IdToToken.Count);
            foreach (var kv in IdToToken)
            {
                bw.Write(kv.Key);
                bw.Write(kv.Value ?? string.Empty);
            }

            // merges
            bw.Write(merges.Count);
            foreach (var (a, b) in merges)
            {
                bw.Write(a ?? string.Empty);
                bw.Write(b ?? string.Empty);
            }

            bw.Write(nextId);
            Console.WriteLine($"Tokenizer sačuvan BIN ({TokenToId.Count} tokena, {merges.Count} spajanja) → {finalPath}");
        }


        public void Load(string path)
        {
            if (!File.Exists(path))
            {
                Console.WriteLine($"Tokenizer fajl nije pronađen: {path}");
                return;
            }

            if (Path.GetExtension(path).Equals(".tokbin", StringComparison.OrdinalIgnoreCase))
            {
                using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
                using var br = new BinaryReader(fs, Encoding.UTF8, leaveOpen: false);

                string magic = br.ReadString();
                if (magic != "TOK") throw new InvalidDataException("Nije TOK bin fajl.");

                int ver = br.ReadInt32();
                if (ver != 1) throw new NotSupportedException($"Nepodržana verzija tokenizera: {ver}");

                // TokenToId
                int t2i = br.ReadInt32();
                TokenToId = new Dictionary<string, int>(t2i);
                for (int i = 0; i < t2i; i++)
                {
                    string k = br.ReadString();
                    int v = br.ReadInt32();
                    TokenToId[k] = v;
                }

                // IdToToken
                int i2t = br.ReadInt32();
                IdToToken = new Dictionary<int, string>(i2t);
                for (int i = 0; i < i2t; i++)
                {
                    int k = br.ReadInt32();
                    string v = br.ReadString();
                    IdToToken[k] = v;
                }

                // merges
                int mc = br.ReadInt32();
                merges = new List<(string, string)>(mc);
                for (int i = 0; i < mc; i++)
                {
                    string a = br.ReadString();
                    string b = br.ReadString();
                    merges.Add((a, b));
                }

                nextId = br.ReadInt32();
                Console.WriteLine($"Tokenizer učitan BIN ({TokenToId.Count} tokena, {merges.Count} spajanja) ← {path}");
            }
            else
            {
                // Back-compat JSON
                var json = File.ReadAllText(path);
                var wrapper = JsonConvert.DeserializeObject<TokenizerData>(json);
                if (wrapper == null) throw new InvalidDataException("Nevalidan JSON tokenizera.");

                TokenToId = wrapper.TokenToId ?? new();
                IdToToken = wrapper.IdToToken ?? new();
                merges = wrapper.Merges ?? new();
                nextId = TokenToId.Count;
                Console.WriteLine(
                    $"Tokenizer učitan JSON ({TokenToId.Count} tokena, {merges.Count} spajanja) ← {path}");
            }
        }

        private class TokenizerData
        {
            public Dictionary<string, int>? TokenToId { get; set; }
            public Dictionary<int, string>? IdToToken { get; set; }
            public List<(string, string)>? Merges { get; set; }
        }
    }
}