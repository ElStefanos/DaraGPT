using System.Text;

namespace DaraGPT.Tokenizer;

public class TokenizerSaveLoad : BpeTokenizer
{
    public string path = "./Models/";

    public TokenizerSaveLoad(Dictionary<string, int> tid, Dictionary<int, string> idt, string path = "./Models/")
    {
        TokenToId = tid;
        IdToToken = idt;
        this.path = path;

        if (!this.path.EndsWith("/"))
            this.path += "/";

        if (!Directory.Exists(path))
            Directory.CreateDirectory(path);
    }

    public void Save(string name)
    {
        var path = this.path + name + "/" + name + ".tokbin";

        if (!Directory.Exists(this.path + name))
            Directory.CreateDirectory(this.path + name);

        using (var fs = new FileStream(path, FileMode.Create))
        using (var bw = new BinaryWriter(fs, Encoding.Unicode))
        {
            bw.Write("TOKBIN");
            bw.Write("TOKID:");

            bw.Write(TokenToId.Count);

            foreach (var kv in TokenToId)
            {
                bw.Write(kv.Key);
                bw.Write(kv.Value);
            }

            bw.Write("IDTOK:");

            bw.Write(IdToToken.Count);

            foreach (var kv in IdToToken)
            {
                bw.Write(kv.Key);
                bw.Write(kv.Value);
            }

            bw.Flush();
            bw.Close();
        }
    }

    public void Load(string name)
    {
        var filePath = path + name + "/" + name + ".tokbin";

        if (!File.Exists(filePath))
        {
            Console.WriteLine("File not found: " + filePath);
            return;
        }


        using (var fs = new FileStream(path + name + "/" + name + ".tokbin", FileMode.Open))
        using (var br = new BinaryReader(fs, Encoding.Unicode))
        {
            var magic = br.ReadString();
            if (magic != "TOKBIN")
                throw new Exception("Invalid tokenizer file header!");

            var tokidHeader = br.ReadString();
            if (tokidHeader != "TOKID:")
                throw new Exception("Corrupted tokenizer section (TOKID missing).");

            var tokCount = br.ReadInt32();

            TokenToId.Clear();

            for (var i = 0; i < tokCount; i++)
            {
                var key = br.ReadString();
                var val = br.ReadInt32();
                TokenToId[key] = val;
            }

            var idtokHeader = br.ReadString();
            if (idtokHeader != "IDTOK:")
                throw new Exception("Corrupted tokenizer section (IDTOK missing).");

            var idCount = br.ReadInt32();

            IdToToken.Clear();

            for (var i = 0; i < idCount; i++)
            {
                var key = br.ReadInt32();
                var val = br.ReadString();
                IdToToken[key] = val;
            }

            Console.WriteLine($"Tokenizer file {filePath} loaded.");
        }
    }

    public void SaveSequence(string fileName = "sequences")
    {
        var fullPath = Path.Combine(path + $"{fileName}/", fileName) + ".seqbin";

        if (!Directory.Exists(path))
            Directory.CreateDirectory(path);

        using (var fs = new FileStream(fullPath, FileMode.Create, FileAccess.Write))
        using (var bw = new BinaryWriter(fs))
        {
            bw.Write("SEQBIN");

            bw.Write(Sequences.Count);

            foreach (var seq in Sequences)
            {
                bw.Write(seq.Count);

                foreach (var id in seq)
                    bw.Write(id);
            }

            bw.Flush();
        }

        Console.WriteLine($"Saved {Sequences.Count} sequences to {fullPath}");
    }

    public List<List<int>> LoadSequence(string fileName = "sequences")
    {
        var fullPath = Path.Combine(path + $"{fileName}/", fileName) + ".seqbin";

        if (!File.Exists(fullPath))
            throw new FileNotFoundException($"Sequence file not found: {fullPath}");

        using (var fs = new FileStream(fullPath, FileMode.Open, FileAccess.Read))
        using (var br = new BinaryReader(fs))
        {
            var magic = br.ReadString();
            if (magic != "SEQBIN")
                throw new Exception("Invalid or corrupted sequence file header.");

            var totalSequences = br.ReadInt32();

            for (var i = 0; i < totalSequences; i++)
            {
                var seqLen = br.ReadInt32();
                var seq = new List<int>(seqLen);

                for (var j = 0; j < seqLen; j++)
                    seq.Add(br.ReadInt32());

                Sequences.Add(seq);
            }
        }

        Console.WriteLine($"Loaded {Sequences.Count} sequences from {fullPath}");
        return Sequences;
    }
}