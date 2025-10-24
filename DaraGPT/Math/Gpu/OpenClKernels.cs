using OpenCL.Net;

namespace DaraGPT.Math.Gpu;

public static class OpenClKernels
{
    public static float[] RunMatMulRM(GpuEngine gpu, float[] A, float[] B, int m, int k, int n)
    {
        var aBuf = Cl.CreateBuffer(gpu.Context,
            MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            new IntPtr(sizeof(float) * A.Length), A, out var err);
        var bBuf = Cl.CreateBuffer(gpu.Context,
            MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            new IntPtr(sizeof(float) * B.Length), B, out err);
        var cBuf = Cl.CreateBuffer(gpu.Context,
            MemFlags.WriteOnly,
            new IntPtr(sizeof(float) * m * n), IntPtr.Zero, out err);

        var kernel = Cl.CreateKernel(gpu.Program, "MatMulRM", out err);
        Cl.SetKernelArg(kernel, 0, aBuf);
        Cl.SetKernelArg(kernel, 1, bBuf);
        Cl.SetKernelArg(kernel, 2, cBuf);
        Cl.SetKernelArg(kernel, 3, m);
        Cl.SetKernelArg(kernel, 4, k);
        Cl.SetKernelArg(kernel, 5, n);

        IntPtr[] gws = { m, n };
        Cl.EnqueueNDRangeKernel(gpu.CommandQueue, kernel, 2, null, gws, null, 0, null, out _);
        Cl.Finish(gpu.CommandQueue);

        var C = new float[m * n];
        Cl.EnqueueReadBuffer(gpu.CommandQueue, cBuf, Bool.True, IntPtr.Zero,
            new IntPtr(sizeof(float) * C.Length), C, 0, null, out _);

        Cl.ReleaseKernel(kernel);
        Cl.ReleaseMemObject(aBuf);
        Cl.ReleaseMemObject(bBuf);
        Cl.ReleaseMemObject(cBuf);
        return C;
    }


    public static float[] RunTranspose2D(GpuEngine gpu, float[] A, int rows, int cols)
    {
        var inBuf = Cl.CreateBuffer(gpu.Context,
            MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            new IntPtr(sizeof(float) * A.Length), A, out var err);
        var outBuf = Cl.CreateBuffer(gpu.Context,
            MemFlags.WriteOnly,
            new IntPtr(sizeof(float) * A.Length), IntPtr.Zero, out err);

        var kernel = Cl.CreateKernel(gpu.Program, "Transpose2D", out err);
        Cl.SetKernelArg(kernel, 0, inBuf);
        Cl.SetKernelArg(kernel, 1, outBuf);
        Cl.SetKernelArg(kernel, 2, rows);
        Cl.SetKernelArg(kernel, 3, cols);

        IntPtr[] gws = { rows, cols };
        Cl.EnqueueNDRangeKernel(gpu.CommandQueue, kernel, 2, null, gws, null, 0, null, out _);
        Cl.Finish(gpu.CommandQueue);

        var result = new float[A.Length];
        Cl.EnqueueReadBuffer(gpu.CommandQueue, outBuf, Bool.True, IntPtr.Zero,
            new IntPtr(sizeof(float) * result.Length), result, 0, null, out _);

        Cl.ReleaseKernel(kernel);
        Cl.ReleaseMemObject(inBuf);
        Cl.ReleaseMemObject(outBuf);
        return result;
    }


    public static void RunScale(GpuEngine gpu, float[] A, float s, int rows, int cols)
    {
        var buf = Cl.CreateBuffer(gpu.Context,
            MemFlags.ReadWrite | MemFlags.CopyHostPtr,
            new IntPtr(sizeof(float) * A.Length), A, out var err);

        var kernel = Cl.CreateKernel(gpu.Program, "ScaleInPlace", out err);
        Cl.SetKernelArg(kernel, 0, buf);
        Cl.SetKernelArg(kernel, 1, s);
        Cl.SetKernelArg(kernel, 2, rows);
        Cl.SetKernelArg(kernel, 3, cols);

        IntPtr[] gws = { rows, cols };
        Cl.EnqueueNDRangeKernel(gpu.CommandQueue, kernel, 2, null, gws, null, 0, null, out _);
        Cl.Finish(gpu.CommandQueue);

        Cl.EnqueueReadBuffer(gpu.CommandQueue, buf, Bool.True, IntPtr.Zero,
            new IntPtr(sizeof(float) * A.Length), A, 0, null, out _);

        Cl.ReleaseKernel(kernel);
        Cl.ReleaseMemObject(buf);
    }


    public static float[] RunRowSoftmax(GpuEngine gpu, float[] A, int rows, int cols)
    {
        var inBuf = Cl.CreateBuffer(gpu.Context,
            MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            new IntPtr(sizeof(float) * A.Length), A, out var err);
        var outBuf = Cl.CreateBuffer(gpu.Context,
            MemFlags.WriteOnly,
            new IntPtr(sizeof(float) * A.Length), IntPtr.Zero, out err);

        var kernel = Cl.CreateKernel(gpu.Program, "RowSoftmax", out err);
        Cl.SetKernelArg(kernel, 0, inBuf);
        Cl.SetKernelArg(kernel, 1, outBuf);
        Cl.SetKernelArg(kernel, 2, rows);
        Cl.SetKernelArg(kernel, 3, cols);

        IntPtr[] gws = { rows };
        Cl.EnqueueNDRangeKernel(gpu.CommandQueue, kernel, 1, null, gws, null, 0, null, out _);
        Cl.Finish(gpu.CommandQueue);

        var result = new float[A.Length];
        Cl.EnqueueReadBuffer(gpu.CommandQueue, outBuf, Bool.True, IntPtr.Zero,
            new IntPtr(sizeof(float) * result.Length), result, 0, null, out _);

        Cl.ReleaseKernel(kernel);
        Cl.ReleaseMemObject(inBuf);
        Cl.ReleaseMemObject(outBuf);
        return result;
    }

    public static float[] RunEmbeddingLookup(GpuEngine gpu, int[] tokenIds, IMem<float> embeddingMatrix, int dModel)
    {
        if (!gpu.Available)
            throw new Exception("GPU not available.");

        if (!gpu.Program.IsValid())
            throw new Exception(
                "GPU program not compiled. Call gpu.LoadKernel() and gpu.CompileKernel() before using.");

        var seqLen = tokenIds.Length;
        var outputSize = seqLen * dModel;
        var output = new float[outputSize];

        var tokenBuffer = Cl.CreateBuffer(
            gpu.Context,
            MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            tokenIds,
            out var err1);

        var outputBuffer = Cl.CreateBuffer<float>(
            gpu.Context,
            MemFlags.WriteOnly,
            outputSize,
            out var err2);

        if (err1 != ErrorCode.Success || err2 != ErrorCode.Success)
            throw new Exception("Failed to create buffers for embedding lookup.");

        var kernel = Cl.CreateKernel(gpu.Program, "EmbeddingLookup", out var err3);
        if (err3 != ErrorCode.Success)
            throw new Exception($"Failed to create EmbeddingLookup kernel: {err3}");

        Cl.SetKernelArg(kernel, 0, tokenBuffer);
        Cl.SetKernelArg(kernel, 1, embeddingMatrix);
        Cl.SetKernelArg(kernel, 2, outputBuffer);
        Cl.SetKernelArg(kernel, 3, dModel);

        var globalWorkSize = new[] { (IntPtr)seqLen };
        var err4 = Cl.EnqueueNDRangeKernel(
            gpu.CommandQueue, kernel, 1,
            null, globalWorkSize, null, 0, null, out _);

        if (err4 != ErrorCode.Success)
            throw new Exception($"Failed to enqueue kernel: {err4}");

        var err5 = Cl.EnqueueReadBuffer(
            gpu.CommandQueue,
            outputBuffer,
            Bool.True,
            IntPtr.Zero,
            new IntPtr(outputSize * sizeof(float)),
            output,
            0, null, out _);

        if (err5 != ErrorCode.Success)
            throw new Exception($"Failed to read buffer: {err5}");

        Cl.ReleaseMemObject(tokenBuffer);
        Cl.ReleaseMemObject(outputBuffer);
        Cl.ReleaseKernel(kernel);

        return output;
    }

    public static float[] RunRowSoftmaxBackward(GpuEngine gpu, float[] A, float[] dA, int rows, int cols)
    {
        var aBuf = Cl.CreateBuffer(gpu.Context, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            new IntPtr(sizeof(float) * A.Length), A, out var err);
        var daBuf = Cl.CreateBuffer(gpu.Context, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            new IntPtr(sizeof(float) * dA.Length), dA, out err);
        var dsBuf = Cl.CreateBuffer(gpu.Context, MemFlags.WriteOnly,
            new IntPtr(sizeof(float) * rows * cols), IntPtr.Zero, out err);

        var kernel = Cl.CreateKernel(gpu.Program, "RowSoftmaxBackward", out err);
        Cl.SetKernelArg(kernel, 0, aBuf);
        Cl.SetKernelArg(kernel, 1, daBuf);
        Cl.SetKernelArg(kernel, 2, dsBuf);
        Cl.SetKernelArg(kernel, 3, rows);
        Cl.SetKernelArg(kernel, 4, cols);

        IntPtr[] gws = { rows };
        Cl.EnqueueNDRangeKernel(gpu.CommandQueue, kernel, 1, null, gws, null, 0, null, out _);
        Cl.Finish(gpu.CommandQueue);

        var dScores = new float[rows * cols];
        Cl.EnqueueReadBuffer(gpu.CommandQueue, dsBuf, Bool.True, IntPtr.Zero,
            new IntPtr(sizeof(float) * dScores.Length), dScores, 0, null, out _);

        Cl.ReleaseKernel(kernel);
        Cl.ReleaseMemObject(aBuf);
        Cl.ReleaseMemObject(daBuf);
        Cl.ReleaseMemObject(dsBuf);
        return dScores;
    }

    public static (float[] Y, float[] MU, float[] INVSTD) RunLayerNormForward(GpuEngine gpu, float[] X, int rows,
        int cols, float eps = 1e-5f)
    {
        var xBuf = Cl.CreateBuffer(gpu.Context, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            new IntPtr(sizeof(float) * X.Length), X, out var err);
        var yBuf = Cl.CreateBuffer(gpu.Context, MemFlags.WriteOnly,
            new IntPtr(sizeof(float) * X.Length), IntPtr.Zero, out err);
        var muBuf = Cl.CreateBuffer(gpu.Context, MemFlags.WriteOnly,
            new IntPtr(sizeof(float) * rows), IntPtr.Zero, out err);
        var isBuf = Cl.CreateBuffer(gpu.Context, MemFlags.WriteOnly,
            new IntPtr(sizeof(float) * rows), IntPtr.Zero, out err);

        var kernel = Cl.CreateKernel(gpu.Program, "LayerNormForward", out err);
        Cl.SetKernelArg(kernel, 0, xBuf);
        Cl.SetKernelArg(kernel, 1, yBuf);
        Cl.SetKernelArg(kernel, 2, muBuf);
        Cl.SetKernelArg(kernel, 3, isBuf);
        Cl.SetKernelArg(kernel, 4, rows);
        Cl.SetKernelArg(kernel, 5, cols);
        Cl.SetKernelArg(kernel, 6, eps);

        IntPtr[] gws = { rows };
        Cl.EnqueueNDRangeKernel(gpu.CommandQueue, kernel, 1, null, gws, null, 0, null, out _);
        Cl.Finish(gpu.CommandQueue);

        var Y = new float[X.Length];
        var MU = new float[rows];
        var INVSTD = new float[rows];
        Cl.EnqueueReadBuffer(gpu.CommandQueue, yBuf, Bool.True, IntPtr.Zero, new IntPtr(sizeof(float) * Y.Length),
            Y, 0, null, out _);
        Cl.EnqueueReadBuffer(gpu.CommandQueue, muBuf, Bool.True, IntPtr.Zero, new IntPtr(sizeof(float) * MU.Length),
            MU, 0, null, out _);
        Cl.EnqueueReadBuffer(gpu.CommandQueue, isBuf, Bool.True, IntPtr.Zero,
            new IntPtr(sizeof(float) * INVSTD.Length), INVSTD, 0, null, out _);

        Cl.ReleaseKernel(kernel);
        Cl.ReleaseMemObject(xBuf);
        Cl.ReleaseMemObject(yBuf);
        Cl.ReleaseMemObject(muBuf);
        Cl.ReleaseMemObject(isBuf);
        return (Y, MU, INVSTD);
    }

    public static float[] RunLayerNormBackward(GpuEngine gpu, float[] X, float[] dY, float[] MU, float[] INVSTD,
        int rows, int cols)
    {
        var xBuf = Cl.CreateBuffer(gpu.Context, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            new IntPtr(sizeof(float) * X.Length), X, out var err);
        var dyBuf = Cl.CreateBuffer(gpu.Context, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            new IntPtr(sizeof(float) * dY.Length), dY, out err);
        var muBuf = Cl.CreateBuffer(gpu.Context, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            new IntPtr(sizeof(float) * MU.Length), MU, out err);
        var isBuf = Cl.CreateBuffer(gpu.Context, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            new IntPtr(sizeof(float) * INVSTD.Length), INVSTD, out err);
        var dxBuf = Cl.CreateBuffer(gpu.Context, MemFlags.WriteOnly,
            new IntPtr(sizeof(float) * rows * cols), IntPtr.Zero, out err);

        var kernel = Cl.CreateKernel(gpu.Program, "LayerNormBackward", out err);
        Cl.SetKernelArg(kernel, 0, xBuf);
        Cl.SetKernelArg(kernel, 1, dyBuf);
        Cl.SetKernelArg(kernel, 2, muBuf);
        Cl.SetKernelArg(kernel, 3, isBuf);
        Cl.SetKernelArg(kernel, 4, dxBuf);
        Cl.SetKernelArg(kernel, 5, rows);
        Cl.SetKernelArg(kernel, 6, cols);

        IntPtr[] gws = { rows };
        Cl.EnqueueNDRangeKernel(gpu.CommandQueue, kernel, 1, null, gws, null, 0, null, out _);
        Cl.Finish(gpu.CommandQueue);

        var dX = new float[rows * cols];
        Cl.EnqueueReadBuffer(gpu.CommandQueue, dxBuf, Bool.True, IntPtr.Zero, new IntPtr(sizeof(float) * dX.Length),
            dX, 0, null, out _);

        Cl.ReleaseKernel(kernel);
        Cl.ReleaseMemObject(xBuf);
        Cl.ReleaseMemObject(dyBuf);
        Cl.ReleaseMemObject(muBuf);
        Cl.ReleaseMemObject(isBuf);
        Cl.ReleaseMemObject(dxBuf);
        return dX;
    }
}