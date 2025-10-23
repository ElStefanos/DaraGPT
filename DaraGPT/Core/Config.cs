using System.IO;

namespace DaraGPT
{
    public class Config
    {
        public int DModel { get; set; } = 1024;
        public int NumLayers { get; set; } = 8;
        public int ContextSize { get; set; } = 500;
        public int VocabSize { get; set; } = 30000;
        public float LearningRate { get; set; } = 1e-4f;
        public string CheckpointDir { get; set; } = "checkpoints";

        public string DevicePreference { get; set; } = "AMD"; // "NVIDIA", "INTEL", "CPU"

        internal void WriteTo(BinaryWriter bw)
        {
            bw.Write(DModel);
            bw.Write(NumLayers);
            bw.Write(ContextSize);
            bw.Write(VocabSize);
            bw.Write(LearningRate);
            bw.Write(CheckpointDir ?? string.Empty);
            bw.Write(DevicePreference ?? string.Empty);
        }

        internal static Config ReadFrom(BinaryReader br)
        {
            return new Config
            {
                DModel        = br.ReadInt32(),
                NumLayers     = br.ReadInt32(),
                ContextSize   = br.ReadInt32(),
                VocabSize     = br.ReadInt32(),
                LearningRate  = br.ReadSingle(),
                CheckpointDir = br.ReadString(),
                DevicePreference = br.ReadString()
            };
        }
    }
}