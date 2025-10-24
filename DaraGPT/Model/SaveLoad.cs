using System.Text;

namespace DaraGPT.Model;

public static class SaveLoad
{
    public static void SaveModel(GPTModel model, string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var fs = new FileStream(path, FileMode.Create);
        using var bw = new BinaryWriter(fs, Encoding.UTF8);

        bw.Write("MODBIN");
        bw.Write(model.Config.VocabSize);
        bw.Write(model.Config.DModel);
        bw.Write(model.Config.Head);
        bw.Write(model.Config.ContextSize);

        var emb = model.Embedding;
        bw.Write(model.Config.VocabSize);
        bw.Write(emb.Dim);
        foreach (var f in emb.Weights)
            bw.Write(f);

        bw.Write(model.Blocks.Length);
        foreach (var block in model.Blocks)
        {
            SaveMatrix(bw, block.Mha.Wq);
            SaveMatrix(bw, block.Mha.Wk);
            SaveMatrix(bw, block.Mha.Wv);
            SaveMatrix(bw, block.Mha.Wo);
            SaveMatrix(bw, block.FfnW1);
            SaveMatrix(bw, block.FfnW2);
        }

        SaveMatrix(bw, model.Head.Weights);

        bw.Flush();
        bw.Close();
    }

    public static void LoadModel(GPTModel model, string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException("Model file not found: " + path);

        using var fs = new FileStream(path, FileMode.Open);
        using var br = new BinaryReader(fs, Encoding.UTF8);

        var magic = new string(br.ReadChars(6));
        if (magic != "MODBIN")
            throw new Exception("Invalid model file format.");

        var vocab = br.ReadInt32();
        var dModel = br.ReadInt32();
        var head = br.ReadInt32();
        var ctx = br.ReadInt32();

        var embVocab = br.ReadInt32();
        var embDim = br.ReadInt32();
        var weights = new float[embVocab * embDim];
        for (var i = 0; i < weights.Length; i++)
            weights[i] = br.ReadSingle();
        model.Embedding.LoadWeights(weights);

        var numBlocks = br.ReadInt32();
        for (var b = 0; b < numBlocks; b++)
        {
            model.Blocks[b].Mha.Wq = LoadMatrix(br);
            model.Blocks[b].Mha.Wk = LoadMatrix(br);
            model.Blocks[b].Mha.Wv = LoadMatrix(br);
            model.Blocks[b].Mha.Wo = LoadMatrix(br);
            model.Blocks[b].FfnW1 = LoadMatrix(br);
            model.Blocks[b].FfnW2 = LoadMatrix(br);
        }

        model.Head.Weights = LoadMatrix(br);
    }

    private static void SaveMatrix(BinaryWriter bw, float[,] matrix)
    {
        var rows = matrix.GetLength(0);
        var cols = matrix.GetLength(1);
        bw.Write(rows);
        bw.Write(cols);
        for (var i = 0; i < rows; i++)
        for (var j = 0; j < cols; j++)
            bw.Write(matrix[i, j]);
    }

    private static float[,] LoadMatrix(BinaryReader br)
    {
        var rows = br.ReadInt32();
        var cols = br.ReadInt32();
        var m = new float[rows, cols];
        for (var i = 0; i < rows; i++)
        for (var j = 0; j < cols; j++)
            m[i, j] = br.ReadSingle();
        return m;
    }
}