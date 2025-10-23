using System.Runtime.InteropServices;
using System.Text;
using Newtonsoft.Json;

// MemoryMarshal

namespace DaraGPT;

[Serializable]
public class GPTModel
{
    private const string MAGIC = "DARA-GPT";
    private const int VERSION = 1;

    [JsonIgnore] public GpuEngine gpu;

    public GPTModel(Config cfg)
    {
        if (cfg == null)
            throw new ArgumentNullException(nameof(cfg), "Config ne sme biti null pri kreiranju modela.");

        this.cfg = cfg;
        gpu = new GpuEngine(cfg.DevicePreference);

        Layers = new List<TransformerLayer>();
        for (var i = 0; i < cfg.NumLayers; i++)
            Layers.Add(new TransformerLayer(cfg.DModel, gpu));

        TokenEmbeddings = new Linear(cfg.VocabSize, cfg.DModel);
        OutputProjection = new Linear(cfg.DModel, cfg.VocabSize);
    }

    public Config cfg { get; private set; }
    public List<TransformerLayer> Layers { get; private set; }
    public Linear TokenEmbeddings { get; set; }
    public Linear OutputProjection { get; private set; }

    //Forward: samo skrivena stanja
    public float[] ForwardHidden(int[] tokens)
    {
        var seq = tokens.Length;
        var inp = new float[seq * cfg.DModel];

        for (var i = 0; i < seq; i++)
        for (var d = 0; d < cfg.DModel; d++)
        {
            var col = Math.Min(tokens[i], TokenEmbeddings.In - 1);
            inp[i * cfg.DModel + d] = TokenEmbeddings.W[d * TokenEmbeddings.In + col];
        }

        var x = inp;
        foreach (var layer in Layers)
            x = layer.Forward(x, seq);

        return x;
    }

    public float[] ProjectToVocab(float[] hidden, int seq)
    {
        return OutputProjection.Forward(hidden, seq, gpu);
    }

    public float[] ForwardTokens(int[] tokens)
    {
        return ProjectToVocab(ForwardHidden(tokens), tokens.Length);
    }

    //Backward na izlazu
    public float[] BackwardOnOutput(float[] hidden, float[] gradLogits, int seq, float lr)
    {
        var dModel = cfg.DModel;
        var vocab = cfg.VocabSize;

        var gradW = new float[vocab * dModel];
        var gradB = new float[vocab];
        var gradIn = new float[seq * dModel];

        if (gpu != null && gpu.Available)
        {
            try
            {
                gpu.LinearBackwardGpu(
                    hidden, seq, dModel,
                    OutputProjection.W, vocab,
                    gradLogits,
                    gradW, gradB, gradIn
                );

                OutputProjection.W = gpu.UpdateWeights(OutputProjection.W, gradW, lr);
                OutputProjection.B = gpu.UpdateWeights(OutputProjection.B, gradB, lr);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️ GPU backward failed, switching to CPU: {ex.Message}");
                gpu.Available = false;
                return BackwardOnOutput(hidden, gradLogits, seq, lr);
            }
        }
        else
        {
            gradIn = LinAlgCPU.MatMul(gradLogits, OutputProjection.W, seq, vocab, dModel);
            gradW = LinAlgCPU.MatMulTranspose(gradLogits, hidden, vocab, seq, dModel);

            for (var i = 0; i < vocab; i++)
            {
                double sum = 0;
                for (var b = 0; b < seq; b++)
                    sum += gradLogits[b * vocab + i];
                gradB[i] = (float)(sum / Math.Max(1, seq));
            }

            for (var i = 0; i < OutputProjection.W.Length; i++)
                OutputProjection.W[i] -= lr * gradW[i];
            for (var i = 0; i < OutputProjection.B.Length; i++)
                OutputProjection.B[i] -= lr * gradB[i];
        }

        return gradIn;
    }

    public void Save(string baseDir)
    {
        Directory.CreateDirectory(baseDir);
        var path = Path.Combine(baseDir, "model.bin");

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var bw = new BinaryWriter(fs, Encoding.UTF8, false);

        // Header
        bw.Write(MAGIC);
        bw.Write(VERSION);

        cfg.WriteTo(bw);

        // Embeddings i Output proj
        WriteLinear(bw, TokenEmbeddings);
        WriteLinear(bw, OutputProjection);

        bw.Write(Layers.Count);
        for (var i = 0; i < Layers.Count; i++)
        {
            var layerJson = JsonConvert.SerializeObject(Layers[i]);
            bw.Write(layerJson);
        }

        bw.Flush();
        Console.WriteLine($"Model sačuvan u: {path}");
    }

    //BINARNO UČITAVANJE (sa back-compat za .json)
    public static GPTModel Load(string path)
    {
        // Back-compat: ako se prosledi folder ili .json
        if (Directory.Exists(path))
        {
            var bin = Path.Combine(path, "model.bin");
            var json = Path.Combine(path, "model.json");
            if (File.Exists(bin)) return Load(bin);
            if (File.Exists(json)) return LoadJson(json); // stari format
            throw new FileNotFoundException($"Nema ni model.bin ni model.json u: {path}");
        }

        // Ako je stari JSON fajl:
        if (string.Equals(Path.GetExtension(path), ".json", StringComparison.OrdinalIgnoreCase))
            return LoadJson(path);

        // Novi binarni format:
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        using var br = new BinaryReader(fs, Encoding.UTF8, false);

        var magic = br.ReadString();
        if (!string.Equals(magic, MAGIC, StringComparison.Ordinal))
            throw new InvalidDataException("Nije DARA-GPT binarni fajl (pogrešna magija).");

        var ver = br.ReadInt32();
        if (ver != VERSION)
            throw new NotSupportedException($"Nepodržana verzija modela: {ver} (očekivano {VERSION}).");

        var cfg = Config.ReadFrom(br);
        var model = new GPTModel(cfg);

        model.TokenEmbeddings = ReadLinear(br);
        model.OutputProjection = ReadLinear(br);

        var nLayers = br.ReadInt32();
        model.Layers.Clear();
        for (var i = 0; i < nLayers; i++)
        {
            // Učitavamo sloj iz JSON bloka (ali unutar binarnog kontejnera)
            var layerJson = br.ReadString();
            var layer = JsonConvert.DeserializeObject<TransformerLayer>(layerJson);
            if (layer == null)
                throw new InvalidDataException($"Neuspelo učitavanje sloja {i} iz JSON bloka.");

            try
            {
                var pi = typeof(TransformerLayer).GetProperty("gpu") ??
                         typeof(TransformerLayer).GetProperty("Gpu") ??
                         typeof(TransformerLayer).GetField("gpu")?.FieldType.GetProperty("gpu");
                // fallback: ako nema, ignoriši; sloj ionako ne koristi GPU direktno u forward-u
            }

            catch
            {
                /* ignore */
            }

            model.Layers.Add(layer);
        }

        Console.WriteLine($"Model učitan uspešno. GPU preferenca: {model.cfg.DevicePreference}");
        return model;
    }

    //Back-compat loader za stari JSON
    private static GPTModel LoadJson(string jsonPath)
    {
        var json = File.ReadAllText(jsonPath);
        var model = JsonConvert.DeserializeObject<GPTModel>(json,
            new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.Auto,
                PreserveReferencesHandling = PreserveReferencesHandling.Objects
            });

        if (model == null)
            throw new Exception("Neuspelo učitavanje modela (deserijalizacija vratila null).");

        if (model.cfg == null)
        {
            Console.WriteLine("Upozorenje: Config nije pronađen u model fajlu. Kreiram podrazumevani...");
            model.cfg = new Config();
        }

        model.gpu = new GpuEngine(model.cfg.DevicePreference);
        Console.WriteLine("[JSON] Model učitan. (preporuka: pređi na model.bin)");
        return model;
    }

    // HELPERI (lokalni, samo za ovaj fajl)
    private static void WriteLinear(BinaryWriter bw, Linear lin)
    {
        bw.Write(lin.In);
        bw.Write(lin.Out);
        WriteFloatArray(bw, lin.W);
        WriteFloatArray(bw, lin.B);
    }

    private static Linear ReadLinear(BinaryReader br)
    {
        var i = br.ReadInt32();
        var o = br.ReadInt32();
        var lin = new Linear(i, o)
        {
            W = ReadFloatArray(br),
            B = ReadFloatArray(br)
        };
        return lin;
    }

    private static void WriteFloatArray(BinaryWriter bw, float[] a)
    {
        bw.Write(a.Length);
        var span = MemoryMarshal.AsBytes(a.AsSpan());
        bw.Write(span);
    }

    private static float[] ReadFloatArray(BinaryReader br)
    {
        var len = br.ReadInt32();
        var arr = new float[len];
        var span = MemoryMarshal.AsBytes(arr.AsSpan());
        var need = span.Length;
        var buf = br.ReadBytes(need);
        if (buf.Length != need) throw new EndOfStreamException("Nepotpuni float[] podaci.");
        buf.AsSpan().CopyTo(span);
        return arr;
    }
}