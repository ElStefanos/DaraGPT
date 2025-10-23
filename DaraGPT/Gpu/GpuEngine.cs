using OpenCL.Net;
using ClProgram = OpenCL.Net.Program;

namespace DaraGPT;

public class GpuEngine : IDisposable
{
    private ClProgram program;

    public GpuEngine(string preferVendor = null)
    {
        try
        {
            var platforms = Cl.GetPlatformIDs(out var err);
            if (err != ErrorCode.Success) return;

            foreach (var plat in platforms)
            {
                var devices = Cl.GetDeviceIDs(plat, DeviceType.Gpu, out err);
                if (err != ErrorCode.Success) continue;

                foreach (var dev in devices)
                {
                    var devName = Cl.GetDeviceInfo(dev, DeviceInfo.Name, out err).ToString();
                    var platName = Cl.GetPlatformInfo(plat, PlatformInfo.Name, out err).ToString();

                    if (preferVendor == null ||
                        devName.ToUpper().Contains(preferVendor.ToUpper()) ||
                        platName.ToUpper().Contains(preferVendor.ToUpper()))
                    {
                        Device = dev;
                        Context = Cl.CreateContext(null, 1, new[] { Device }, null, IntPtr.Zero, out err);
                        Queue = Cl.CreateCommandQueue(Context, Device, 0, out err);
                        Available = true;
                        BuildProgram();
                        Console.WriteLine($"GPU initialized: {devName} ({platName})");
                        return;
                    }
                }
            }

            Console.WriteLine("No suitable GPU found, falling back to CPU.");
        }
        catch (Exception ex)
        {
            Console.WriteLine("OpenCL init failed: " + ex.Message);
        }
    }

    public Device Device { get; }
    public Context Context { get; }
    public CommandQueue Queue { get; }
    public bool Available { get; set; }

    public void Dispose()
    {
        try
        {
            if (!Queue.Equals(default(CommandQueue)))
                Cl.ReleaseCommandQueue(Queue);
            if (!program.Equals(default(ClProgram)))
                Cl.ReleaseProgram(program);
            if (!Context.Equals(default(Context)))
                Cl.ReleaseContext(Context);
        }
        catch
        {
            Console.WriteLine("Failed to release context.");
        }
    }

    private void BuildProgram()
    {
        // Svi potrebni kerneli
        var src = File.ReadAllText("./GPUKernel/Kernel.txt");

        try
        {
            var err = ErrorCode.Success;
            program = Cl.CreateProgramWithSource(Context, 1, new[] { src }, null, out err);
            if (err != ErrorCode.Success)
                throw new Exception($"CreateProgramWithSource failed: {err}");

            err = Cl.BuildProgram(program, 0, null, "-cl-fast-relaxed-math", null, IntPtr.Zero);

            if (err != ErrorCode.Success)
            {
                var log = Cl.GetProgramBuildInfo(program, Device, ProgramBuildInfo.Log, out _).ToString();
                Console.WriteLine("OpenCL build log:\n" + log);
                throw new Exception("OpenCL program build failed: " + err);
            }

            Console.WriteLine("OpenCL kernels compiled successfully.");
        }
        catch (Exception ex)
        {
            Console.WriteLine("GPU kernel build failed: " + ex.Message);
            Available = false;
        }
    }

    public float[] MatMul(float[] A, float[] B, int M, int N, int P)
    {
        if (!Available) return LinAlgCPU.MatMul(A, B, M, N, P);

        var err = ErrorCode.Success;
        var bufA = Cl.CreateBuffer(Context, MemFlags.ReadOnly | MemFlags.CopyHostPtr, A.Length * sizeof(float), A,
            out err);
        var bufB = Cl.CreateBuffer(Context, MemFlags.ReadOnly | MemFlags.CopyHostPtr, B.Length * sizeof(float), B,
            out err);
        var bufOut = Cl.CreateBuffer(Context, MemFlags.WriteOnly, M * P * sizeof(float), out err);

        var kernel = Cl.CreateKernel(program, "MatMulOptimized", out err);
        err |= Cl.SetKernelArg(kernel, 0, M);
        err |= Cl.SetKernelArg(kernel, 1, N);
        err |= Cl.SetKernelArg(kernel, 2, P);
        err |= Cl.SetKernelArg(kernel, 3, bufA);
        err |= Cl.SetKernelArg(kernel, 4, bufB);
        err |= Cl.SetKernelArg(kernel, 5, bufOut);

        var global = new[] { M, (IntPtr)P };
        err |= Cl.EnqueueNDRangeKernel(Queue, kernel, 2, null, global, null, 0, null, out _);

        var result = new float[M * P];
        err |= Cl.EnqueueReadBuffer(Queue, bufOut, Bool.True, IntPtr.Zero, new IntPtr(result.Length * sizeof(float)),
            result, 0, null, out _);
        Cl.Finish(Queue);

        Cl.ReleaseKernel(kernel);
        Cl.ReleaseMemObject(bufA);
        Cl.ReleaseMemObject(bufB);
        Cl.ReleaseMemObject(bufOut);

        return result;
    }

    public void LinearBackwardGpu(
        float[] X, int B, int In,
        float[] W, int Out,
        float[] GradOut,
        float[] GradW, float[] GradB, float[] GradIn)
    {
        if (!Available)
            throw new InvalidOperationException("GPU not available.");

        var err = ErrorCode.Success;

        var bufX = Cl.CreateBuffer(Context, MemFlags.ReadOnly | MemFlags.CopyHostPtr, X.Length * sizeof(float), X,
            out err);
        var bufW = Cl.CreateBuffer(Context, MemFlags.ReadOnly | MemFlags.CopyHostPtr, W.Length * sizeof(float), W,
            out err);
        var bufGO = Cl.CreateBuffer(Context, MemFlags.ReadOnly | MemFlags.CopyHostPtr, GradOut.Length * sizeof(float),
            GradOut, out err);

        var bufGradW = Cl.CreateBuffer(Context, MemFlags.WriteOnly, GradW.Length * sizeof(float), out err);
        var bufGradB = Cl.CreateBuffer(Context, MemFlags.WriteOnly, GradB.Length * sizeof(float), out err);
        var bufGradIn = Cl.CreateBuffer(Context, MemFlags.WriteOnly, GradIn.Length * sizeof(float), out err);

        var kW = Cl.CreateKernel(program, "LinearGradW", out err);
        Cl.SetKernelArg(kW, 0, B);
        Cl.SetKernelArg(kW, 1, In);
        Cl.SetKernelArg(kW, 2, Out);
        Cl.SetKernelArg(kW, 3, bufX);
        Cl.SetKernelArg(kW, 4, bufGO);
        Cl.SetKernelArg(kW, 5, bufGradW);
        var globalW = new[] { Out, (IntPtr)In };
        Cl.EnqueueNDRangeKernel(Queue, kW, 2, null, globalW, null, 0, null, out _);
        Cl.ReleaseKernel(kW);

        var kB = Cl.CreateKernel(program, "LinearGradB", out err);
        Cl.SetKernelArg(kB, 0, B);
        Cl.SetKernelArg(kB, 1, Out);
        Cl.SetKernelArg(kB, 2, bufGO);
        Cl.SetKernelArg(kB, 3, bufGradB);
        var globalB = new[] { (IntPtr)Out };
        Cl.EnqueueNDRangeKernel(Queue, kB, 1, null, globalB, null, 0, null, out _);
        Cl.ReleaseKernel(kB);

        var kIn = Cl.CreateKernel(program, "LinearGradInput", out err);
        Cl.SetKernelArg(kIn, 0, B);
        Cl.SetKernelArg(kIn, 1, In);
        Cl.SetKernelArg(kIn, 2, Out);
        Cl.SetKernelArg(kIn, 3, bufW);
        Cl.SetKernelArg(kIn, 4, bufGO);
        Cl.SetKernelArg(kIn, 5, bufGradIn);
        var globalIn = new[] { B, (IntPtr)In };
        Cl.EnqueueNDRangeKernel(Queue, kIn, 2, null, globalIn, null, 0, null, out _);
        Cl.ReleaseKernel(kIn);

        Cl.EnqueueReadBuffer(Queue, bufGradW, Bool.True, IntPtr.Zero, new IntPtr(GradW.Length * sizeof(float)), GradW,
            0, null, out _);
        Cl.EnqueueReadBuffer(Queue, bufGradB, Bool.True, IntPtr.Zero, new IntPtr(GradB.Length * sizeof(float)), GradB,
            0, null, out _);
        Cl.EnqueueReadBuffer(Queue, bufGradIn, Bool.True, IntPtr.Zero, new IntPtr(GradIn.Length * sizeof(float)),
            GradIn, 0, null, out _);

        Cl.Finish(Queue);

        Cl.ReleaseMemObject(bufX);
        Cl.ReleaseMemObject(bufW);
        Cl.ReleaseMemObject(bufGO);
        Cl.ReleaseMemObject(bufGradW);
        Cl.ReleaseMemObject(bufGradB);
        Cl.ReleaseMemObject(bufGradIn);
    }

    public float[] RowSoftmax(float[] X, int rows, int cols)
    {
        if (!Available) return LinAlgCPU.SoftmaxRowwise(X, rows, cols);

        var err = ErrorCode.Success;
        var bufX = Cl.CreateBuffer(Context, MemFlags.ReadOnly | MemFlags.CopyHostPtr, X.Length * sizeof(float), X,
            out err);
        var bufOut = Cl.CreateBuffer(Context, MemFlags.WriteOnly, X.Length * sizeof(float), out err);

        var kernel = Cl.CreateKernel(program, "RowSoftmaxOptimized", out err);
        err |= Cl.SetKernelArg(kernel, 0, rows);
        err |= Cl.SetKernelArg(kernel, 1, cols);
        err |= Cl.SetKernelArg(kernel, 2, bufX);
        err |= Cl.SetKernelArg(kernel, 3, bufOut);

        var global = new[] { (IntPtr)rows };
        err |= Cl.EnqueueNDRangeKernel(Queue, kernel, 1, null, global, null, 0, null, out _);

        var result = new float[X.Length];
        err |= Cl.EnqueueReadBuffer(Queue, bufOut, Bool.True, IntPtr.Zero, new IntPtr(result.Length * sizeof(float)),
            result, 0, null, out _);
        Cl.Finish(Queue);

        Cl.ReleaseKernel(kernel);
        Cl.ReleaseMemObject(bufX);
        Cl.ReleaseMemObject(bufOut);

        return result;
    }

    public float[] UpdateWeights(float[] weights, float[] grads, float lr)
    {
        if (!Available)
        {
            for (var i = 0; i < weights.Length; i++)
                weights[i] -= lr * grads[i];
            return weights;
        }

        var err = ErrorCode.Success;
        var kernel = Cl.CreateKernel(program, "UpdateWeightsKernel", out err);

        var bufW = Cl.CreateBuffer(Context, MemFlags.ReadWrite | MemFlags.CopyHostPtr,
            weights.Length * sizeof(float), weights, out err);
        var bufGrad = Cl.CreateBuffer(Context, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            grads.Length * sizeof(float), grads, out err);

        Cl.SetKernelArg(kernel, 0, bufW);
        Cl.SetKernelArg(kernel, 1, bufGrad);
        Cl.SetKernelArg(kernel, 2, lr);

        var global = new[] { (IntPtr)weights.Length };
        Cl.EnqueueNDRangeKernel(Queue, kernel, 1, null, global, null, 0, null, out _);

        Cl.EnqueueReadBuffer(Queue, bufW, Bool.True, IntPtr.Zero, new IntPtr(weights.Length * sizeof(float)),
            weights, 0, null, out _);
        Cl.Finish(Queue);

        Cl.ReleaseKernel(kernel);
        Cl.ReleaseMemObject(bufW);
        Cl.ReleaseMemObject(bufGrad);

        return weights;
    }
}