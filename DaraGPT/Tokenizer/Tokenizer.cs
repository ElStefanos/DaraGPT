namespace DaraGPT.Tokenizer;

public class Tokenizer
{
    private readonly string _name;


    public Tokenizer(string name)
    {
        BTokenizer = new BpeTokenizer();
        SaveLoad = new TokenizerSaveLoad(BTokenizer.TokenToId, BTokenizer.IdToToken);
        _name = name;
    }

    public BpeTokenizer BTokenizer { get; set; }
    private TokenizerSaveLoad SaveLoad { get; }


    public BpeTokenizer RunTokenizer(string filePath, int targetVocab = 5000, int contextSize = 512)
    {
        var savePath = SaveLoad.path + _name + "/" + _name + ".tokbin";

        List<List<int>> seq;

        if (File.Exists(savePath))
        {
            Console.WriteLine($"Tokenizer file {savePath} already exists.");
            Console.Write("Do you want to overwrite existing file? [Y/n]: ");
            var answer = Console.ReadLine()?.ToUpper();

            if (answer == "N")
            {
                SaveLoad.Load(_name);

                if (SaveLoad.IdToToken.Count == 0) return BTokenizer;
                BTokenizer.TokenToId = SaveLoad.TokenToId;
                BTokenizer.IdToToken = SaveLoad.IdToToken;

                SaveLoad.LoadSequence(_name);
                BTokenizer.Sequences = SaveLoad.Sequences;

                return BTokenizer;
            }
        }

        if (!File.Exists(filePath))
            throw new FileNotFoundException($"File {filePath} not found.");

        var text = File.ReadAllText(filePath);

        BTokenizer.TrainBpe(new List<string> { text }, targetVocab);

        Console.WriteLine($"Tokenizer file {savePath} has been successfully trained.");
        Console.WriteLine("Encoding text...");

        BTokenizer.Encode(text);

        Console.WriteLine("Creating a sequnce");


        BTokenizer.CreateSequences(new List<string> { text }, contextSize);

        SaveLoad.TokenToId = BTokenizer.TokenToId;
        SaveLoad.IdToToken = BTokenizer.IdToToken;
        SaveLoad.Sequences = BTokenizer.Sequences;

        SaveLoad.Save(_name);
        SaveLoad.SaveSequence(_name);

        return BTokenizer;
    }
}