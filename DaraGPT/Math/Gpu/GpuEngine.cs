using OpenCL.Net;

namespace DaraGPT.Math.Gpu;

public class GpuEngine : IDisposable
{
    private readonly List<string> KernelFiles = new()
    {
        "./GPUKernels/EmbeddingLookup.cl",
        "./GPUKernels/MatMulRM.cl",
        "./GPUKernels/RoPE.cl",
        "./GPUKernels/RowSoftMax.cl",
        "./GPUKernels/ScaleInPlace.cl",
        "./GPUKernels/Transpose2D.cl",
        "./GPUKernels/LayerNorm.cl"
    };

    private string KernelSource = "";

    public GpuEngine(string preferredVendor = null)
    {
        try
        {
            var platforms = Cl.GetPlatformIDs(out var err);
            if (err != ErrorCode.Success)
                throw new Exception("No OpenCL platform found.");

            foreach (var platform in platforms)
            {
                var devices = Cl.GetDeviceIDs(platform, DeviceType.Gpu, out err);
                if (err != ErrorCode.Success) continue;

                foreach (var device in devices)
                {
                    var vendor = Cl.GetDeviceInfo(device, DeviceInfo.Vendor, out _).ToString();
                    var name = Cl.GetDeviceInfo(device, DeviceInfo.Name, out _).ToString();

                    if (preferredVendor == null || vendor.ToUpper().Contains(preferredVendor.ToUpper()))
                    {
                        Device = device;
                        Context = Cl.CreateContext(null, 1, new[] { Device }, null, IntPtr.Zero, out err);
                        CommandQueue = Cl.CreateCommandQueue(Context, Device, 0, out err);
                        Available = true;

                        Console.WriteLine($"[GPU] Initialized: {vendor} {name}");
                        return;
                    }
                }
            }

            Console.WriteLine("[GPU] No compatible device found.");
        }
        catch (Exception ex)
        {
            Console.WriteLine("[GPU] Initialization failed: " + ex.Message);
        }
    }

    public Context Context { get; }
    public Device Device { get; }
    public CommandQueue CommandQueue { get; }
    public OpenCL.Net.Program Program { get; private set; }

    public bool Available { get; private set; }

    public void Dispose()
    {
        try
        {
            if (CommandQueue.IsValid())
                Cl.ReleaseCommandQueue(CommandQueue);

            if (Program.IsValid())
                Cl.ReleaseProgram(Program);

            if (Context.IsValid())
                Cl.ReleaseContext(Context);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[GPU] Dispose warning: {ex.Message}");
        }
        finally
        {
            Available = false;
        }
    }

    /// <summary>
    ///     Loads kernel source code from all files in the KernelFiles list.
    /// </summary>
    public void LoadKernel()
    {
        KernelSource = "";

        foreach (var kernel in KernelFiles)
        {
            if (!File.Exists(kernel))
                throw new FileNotFoundException("Kernel file not found: " + kernel);

            KernelSource += File.ReadAllText(kernel) + "\n";
        }

        Console.WriteLine($"[GPU] Loaded {KernelFiles.Count} kernel file(s).");
    }

    /// <summary>
    ///     Compiles loaded OpenCL kernels into a program.
    /// </summary>
    public void CompileKernel()
    {
        if (!Available)
            throw new Exception("GPU not initialized.");
        if (string.IsNullOrEmpty(KernelSource))
            throw new Exception("No kernel source loaded. Call LoadKernel() first.");

        Program = Cl.CreateProgramWithSource(Context, 1, new[] { KernelSource }, null, out var err);
        err = Cl.BuildProgram(Program, 1, new[] { Device }, string.Empty, null, IntPtr.Zero);

        if (err != ErrorCode.Success)
        {
            var log = Cl.GetProgramBuildInfo(Program, Device, ProgramBuildInfo.Log, out _);
            throw new Exception($"Kernel build failed:\n{log}");
        }

        Console.WriteLine("[GPU] Kernel compiled successfully.");
    }
}