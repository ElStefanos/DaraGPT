using DaraGPT.Math.Gpu;
using OpenCL.Net;

namespace DaraGPT.Model;

public class RotaryEmbedding
{
    private readonly int dModel;
    private readonly GpuEngine gpu;

    public RotaryEmbedding(int dModel, GpuEngine gpu = null)
    {
        this.dModel = dModel;
        this.gpu = gpu;
    }

    public float[,] Apply(float[,] x)
    {
        var seqLen = x.GetLength(0);
        var dim = x.GetLength(1);
        var result = new float[seqLen, dim];

        for (var pos = 0; pos < seqLen; pos++)
        for (var i = 0; i < dim; i += 2)
        {
            var angle = pos / MathF.Pow(10000f, 2f * i / dim);
            var c = MathF.Cos(angle);
            var s = MathF.Sin(angle);

            var even = x[pos, i];
            var odd = i + 1 < dim ? x[pos, i + 1] : 0f;

            result[pos, i] = even * c - odd * s;
            if (i + 1 < dim)
                result[pos, i + 1] = even * s + odd * c;
        }

        return result;
    }

    public void ApplyGpu(float[] tensor, int rows)
    {
        if (gpu == null || !gpu.Available)
            throw new Exception("GPU not initialized for RoPE.");

        var buf = Cl.CreateBuffer(
            gpu.Context,
            MemFlags.ReadWrite | MemFlags.CopyHostPtr,
            tensor,
            out var err);

        if (err != ErrorCode.Success)
            throw new Exception("RoPE: failed to create buffer.");

        var kernel = Cl.CreateKernel(gpu.Program, "RotaryEmbedding", out err);
        if (err != ErrorCode.Success)
            throw new Exception("RoPE: failed to create kernel.");

        Cl.SetKernelArg(kernel, 0, buf);
        Cl.SetKernelArg(kernel, 1, rows);
        Cl.SetKernelArg(kernel, 2, dModel);

        IntPtr[] gws = { rows };
        Cl.EnqueueNDRangeKernel(gpu.CommandQueue, kernel, 1, null, gws, null, 0, null, out _);
        Cl.Finish(gpu.CommandQueue);

        Cl.EnqueueReadBuffer(gpu.CommandQueue, buf, Bool.True, IntPtr.Zero,
            new IntPtr(sizeof(float) * tensor.Length), tensor, 0, null, out _);

        Cl.ReleaseKernel(kernel);
        Cl.ReleaseMemObject(buf);
    }
}