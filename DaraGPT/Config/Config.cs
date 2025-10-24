using Newtonsoft.Json;

namespace DaraGPT.Config;

[Serializable]
public class Config
{
    public Config(string modelName = "DaraGPT", string modelSavePath = "./Model/", string dataPath = "./Data/",
        int dModel = 192, int head = 4, int layers = 6, int contextSize = 512, int vocabSize = 5000,
        float learningRate = 0.0025f, string devicePreference = "INTEL")
    {
        ModelName = modelName;
        ModelSavePath = modelSavePath;
        DataPath = dataPath;
        DModel = dModel;
        Head = head;
        Layers = layers;
        ContextSize = contextSize;
        VocabSize = vocabSize;
        LearningRate = learningRate;
        DevicePreference = devicePreference;
    }

    public string ModelName { get; set; }
    public string ModelSavePath { get; set; }
    public string DataPath { get; set; }
    public int DModel { get; set; }
    public int Head { get; set; }
    public int Layers { get; set; }
    public int ContextSize { get; set; }
    public int VocabSize { get; set; }
    public float LearningRate { get; set; }
    public string DevicePreference { get; set; }

    public static Config LoadOrCreate(string path = "./config.json")
    {
        if (!File.Exists(path))
        {
            var cfg = new Config();
            cfg.Save(path);
            return cfg;
        }

        var json = File.ReadAllText(path);
        return JsonConvert.DeserializeObject<Config>(json);
    }

    public void Save(string path = "./config.json")
    {
        var json = JsonConvert.SerializeObject(this, Formatting.Indented);
        File.WriteAllText(path, json);
    }
}