using System.Runtime.InteropServices;
using System.Text;

namespace DaraGPT;

internal static class BinIO
{
    private static readonly UTF8Encoding Utf8NoBOM = new(false);

    public static void WriteString(BinaryWriter bw, string? s)
    {
        if (s is null)
        {
            bw.Write(-1);
            return;
        }

        var bytes = Utf8NoBOM.GetBytes(s);
        bw.Write(bytes.Length);
        bw.Write(bytes);
    }

    public static string? ReadString(BinaryReader br)
    {
        var len = br.ReadInt32();
        if (len < 0) return null;
        var bytes = br.ReadBytes(len);
        return Utf8NoBOM.GetString(bytes);
    }

    public static void WriteFloatArray(BinaryWriter bw, float[]? a)
    {
        if (a is null)
        {
            bw.Write(-1);
            return;
        }

        bw.Write(a.Length);
        var span = MemoryMarshal.AsBytes(a.AsSpan());
        bw.Write(span);
    }

    public static float[]? ReadFloatArray(BinaryReader br)
    {
        var len = br.ReadInt32();
        if (len < 0) return null;
        var arr = new float[len];
        var span = MemoryMarshal.AsBytes(arr.AsSpan());
        var need = span.Length;
        var buf = br.ReadBytes(need);
        if (buf.Length != need) throw new EndOfStreamException("Nepotpuni float[] podaci.");
        buf.AsSpan().CopyTo(span);
        return arr;
    }

    public static void WriteIntArray(BinaryWriter bw, int[]? a)
    {
        if (a is null)
        {
            bw.Write(-1);
            return;
        }

        bw.Write(a.Length);
        var bytes = new byte[a.Length * sizeof(int)];
        Buffer.BlockCopy(a, 0, bytes, 0, bytes.Length);
        bw.Write(bytes);
    }

    public static int[]? ReadIntArray(BinaryReader br)
    {
        var len = br.ReadInt32();
        if (len < 0) return null;
        var bytes = br.ReadBytes(len * sizeof(int));
        if (bytes.Length != len * sizeof(int)) throw new EndOfStreamException("Nepotpuni int[] podaci.");
        var arr = new int[len];
        Buffer.BlockCopy(bytes, 0, arr, 0, bytes.Length);
        return arr;
    }
}