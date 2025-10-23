using System.IO;

namespace DaraGPT
{
    public interface IBinaryLayer
    {
        void WriteTo(BinaryWriter bw);
    }
}