using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace CSCuda
{
    using cudaUUID_t = CUuuid_st;

    public class DriverTypes
    {
        public const uint cudaHostAllocDefault = 0x00;  /**< Default page-locked allocation flag */
        public const uint cudaHostAllocPortable = 0x01;  /**< Pinned memory accessible by all CUDA contexts */
        public const uint cudaHostAllocMapped = 0x02;  /**< Map allocation into device space */
        public const uint cudaHostAllocWriteCombined = 0x04;  /**< Write-combined memory */

        public const uint cudaHostRegisterDefault = 0x00;  /**< Default host memory registration flag */
        public const uint cudaHostRegisterPortable = 0x01;  /**< Pinned memory accessible by all CUDA contexts */
        public const uint cudaHostRegisterMapped = 0x02;  /**< Map registered memory into device space */
        public const uint cudaHostRegisterIoMemory = 0x04;  /**< Memory-mapped I/O space */

        public const uint cudaPeerAccessDefault = 0x00;  /**< Default peer addressing enable flag */

        public const uint cudaStreamDefault = 0x00;  /**< Default stream flag */
        public const uint cudaStreamNonBlocking = 0x01;  /**< Stream does not synchronize with stream 0 (the NULL stream) */
    }

    /// <summary>
    /// CUDA error types
    /// </summary>
    public enum cudaError
    {
        /**
         * The API call returned with no errors. In the case of query calls, this
         * can also mean that the operation being queried is complete (see
         * ::cudaEventQuery() and ::cudaStreamQuery()).
         */
        cudaSuccess = 0,

        /**
         * The device function being invoked (usually via ::cudaLaunchKernel()) was not
         * previously configured via the ::cudaConfigureCall() function.
         */
        cudaErrorMissingConfiguration = 1,

        /**
         * The API call failed because it was unable to allocate enough memory to
         * perform the requested operation.
         */
        cudaErrorMemoryAllocation = 2,

        /**
         * The API call failed because the CUDA driver and runtime could not be
         * initialized.
         */
        cudaErrorInitializationError = 3,

        /**
         * An exception occurred on the device while executing a kernel. Common
         * causes include dereferencing an invalid device pointer and accessing
         * out of bounds shared memory. The device cannot be used until
         * ::cudaThreadExit() is called. All existing device memory allocations
         * are invalid and must be reconstructed if the program is to continue
         * using CUDA.
         */
        cudaErrorLaunchFailure = 4,

        /**
         * This indicated that a previous kernel launch failed. This was previously
         * used for device emulation of kernel launches.
         * \deprecated
         * This error return is deprecated as of CUDA 3.1. Device emulation mode was
         * removed with the CUDA 3.1 release.
         */
        cudaErrorPriorLaunchFailure = 5,

        /**
         * This indicates that the device kernel took too long to execute. This can
         * only occur if timeouts are enabled - see the device property
         * \ref ::cudaDeviceProp::kernelExecTimeoutEnabled "kernelExecTimeoutEnabled"
         * for more information. The device cannot be used until ::cudaThreadExit()
         * is called. All existing device memory allocations are invalid and must be
         * reconstructed if the program is to continue using CUDA.
         */
        cudaErrorLaunchTimeout = 6,

        /**
         * This indicates that a launch did not occur because it did not have
         * appropriate resources. Although this error is similar to
         * ::cudaErrorInvalidConfiguration, this error usually indicates that the
         * user has attempted to pass too many arguments to the device kernel, or the
         * kernel launch specifies too many threads for the kernel's register count.
         */
        cudaErrorLaunchOutOfResources = 7,

        /**
         * The requested device function does not exist or is not compiled for the
         * proper device architecture.
         */
        cudaErrorInvalidDeviceFunction = 8,

        /**
         * This indicates that a kernel launch is requesting resources that can
         * never be satisfied by the current device. Requesting more shared memory
         * per block than the device supports will trigger this error, as will
         * requesting too many threads or blocks. See ::cudaDeviceProp for more
         * device limitations.
         */
        cudaErrorInvalidConfiguration = 9,

        /**
         * This indicates that the device ordinal supplied by the user does not
         * correspond to a valid CUDA device.
         */
        cudaErrorInvalidDevice = 10,

        /**
         * This indicates that one or more of the parameters passed to the API call
         * is not within an acceptable range of values.
         */
        cudaErrorInvalidValue = 11,

        /**
         * This indicates that one or more of the pitch-related parameters passed
         * to the API call is not within the acceptable range for pitch.
         */
        cudaErrorInvalidPitchValue = 12,

        /**
         * This indicates that the symbol name/identifier passed to the API call
         * is not a valid name or identifier.
         */
        cudaErrorInvalidSymbol = 13,

        /**
         * This indicates that the buffer object could not be mapped.
         */
        cudaErrorMapBufferObjectFailed = 14,

        /**
         * This indicates that the buffer object could not be unmapped.
         */
        cudaErrorUnmapBufferObjectFailed = 15,

        /**
         * This indicates that at least one host pointer passed to the API call is
         * not a valid host pointer.
         */
        cudaErrorInvalidHostPointer = 16,

        /**
         * This indicates that at least one device pointer passed to the API call is
         * not a valid device pointer.
         */
        cudaErrorInvalidDevicePointer = 17,

        /**
         * This indicates that the texture passed to the API call is not a valid
         * texture.
         */
        cudaErrorInvalidTexture = 18,

        /**
         * This indicates that the texture binding is not valid. This occurs if you
         * call ::cudaGetTextureAlignmentOffset() with an unbound texture.
         */
        cudaErrorInvalidTextureBinding = 19,

        /**
         * This indicates that the channel descriptor passed to the API call is not
         * valid. This occurs if the format is not one of the formats specified by
         * ::cudaChannelFormatKind, or if one of the dimensions is invalid.
         */
        cudaErrorInvalidChannelDescriptor = 20,

        /**
         * This indicates that the direction of the memcpy passed to the API call is
         * not one of the types specified by ::cudaMemcpyKind.
         */
        cudaErrorInvalidMemcpyDirection = 21,

        /**
         * This indicated that the user has taken the address of a constant variable,
         * which was forbidden up until the CUDA 3.1 release.
         * \deprecated
         * This error return is deprecated as of CUDA 3.1. Variables in constant
         * memory may now have their address taken by the runtime via
         * ::cudaGetSymbolAddress().
         */
        cudaErrorAddressOfConstant = 22,

        /**
         * This indicated that a texture fetch was not able to be performed.
         * This was previously used for device emulation of texture operations.
         * \deprecated
         * This error return is deprecated as of CUDA 3.1. Device emulation mode was
         * removed with the CUDA 3.1 release.
         */
        cudaErrorTextureFetchFailed = 23,

        /**
         * This indicated that a texture was not bound for access.
         * This was previously used for device emulation of texture operations.
         * \deprecated
         * This error return is deprecated as of CUDA 3.1. Device emulation mode was
         * removed with the CUDA 3.1 release.
         */
        cudaErrorTextureNotBound = 24,

        /**
         * This indicated that a synchronization operation had failed.
         * This was previously used for some device emulation functions.
         * \deprecated
         * This error return is deprecated as of CUDA 3.1. Device emulation mode was
         * removed with the CUDA 3.1 release.
         */
        cudaErrorSynchronizationError = 25,

        /**
         * This indicates that a non-float texture was being accessed with linear
         * filtering. This is not supported by CUDA.
         */
        cudaErrorInvalidFilterSetting = 26,

        /**
         * This indicates that an attempt was made to read a non-float texture as a
         * normalized float. This is not supported by CUDA.
         */
        cudaErrorInvalidNormSetting = 27,

        /**
         * Mixing of device and device emulation code was not allowed.
         * \deprecated
         * This error return is deprecated as of CUDA 3.1. Device emulation mode was
         * removed with the CUDA 3.1 release.
         */
        cudaErrorMixedDeviceExecution = 28,

        /**
         * This indicates that a CUDA Runtime API call cannot be executed because
         * it is being called during process shut down, at a point in time after
         * CUDA driver has been unloaded.
         */
        cudaErrorCudartUnloading = 29,

        /**
         * This indicates that an unknown internal error has occurred.
         */
        cudaErrorUnknown = 30,

        /**
         * This indicates that the API call is not yet implemented. Production
         * releases of CUDA will never return this error.
         * \deprecated
         * This error return is deprecated as of CUDA 4.1.
         */
        cudaErrorNotYetImplemented = 31,

        /**
         * This indicated that an emulated device pointer exceeded the 32-bit address
         * range.
         * \deprecated
         * This error return is deprecated as of CUDA 3.1. Device emulation mode was
         * removed with the CUDA 3.1 release.
         */
        cudaErrorMemoryValueTooLarge = 32,

        /**
         * This indicates that a resource handle passed to the API call was not
         * valid. Resource handles are opaque types like ::cudaStream_t and
         * ::cudaEvent_t.
         */
        cudaErrorInvalidResourceHandle = 33,

        /**
         * This indicates that asynchronous operations issued previously have not
         * completed yet. This result is not actually an error, but must be indicated
         * differently than ::cudaSuccess (which indicates completion). Calls that
         * may return this value include ::cudaEventQuery() and ::cudaStreamQuery().
         */
        cudaErrorNotReady = 34,

        /**
         * This indicates that the installed NVIDIA CUDA driver is older than the
         * CUDA runtime library. This is not a supported configuration. Users should
         * install an updated NVIDIA display driver to allow the application to run.
         */
        cudaErrorInsufficientDriver = 35,

        /**
         * This indicates that the user has called ::cudaSetValidDevices(),
         * ::cudaSetDeviceFlags(), ::cudaD3D9SetDirect3DDevice(),
         * ::cudaD3D10SetDirect3DDevice, ::cudaD3D11SetDirect3DDevice(), or
         * ::cudaVDPAUSetVDPAUDevice() after initializing the CUDA runtime by
         * calling non-device management operations (allocating memory and
         * launching kernels are examples of non-device management operations).
         * This error can also be returned if using runtime/driver
         * interoperability and there is an existing ::CUcontext active on the
         * host thread.
         */
        cudaErrorSetOnActiveProcess = 36,

        /**
         * This indicates that the surface passed to the API call is not a valid
         * surface.
         */
        cudaErrorInvalidSurface = 37,

        /**
         * This indicates that no CUDA-capable devices were detected by the installed
         * CUDA driver.
         */
        cudaErrorNoDevice = 38,

        /**
         * This indicates that an uncorrectable ECC error was detected during
         * execution.
         */
        cudaErrorECCUncorrectable = 39,

        /**
         * This indicates that a link to a shared object failed to resolve.
         */
        cudaErrorSharedObjectSymbolNotFound = 40,

        /**
         * This indicates that initialization of a shared object failed.
         */
        cudaErrorSharedObjectInitFailed = 41,

        /**
         * This indicates that the ::cudaLimit passed to the API call is not
         * supported by the active device.
         */
        cudaErrorUnsupportedLimit = 42,

        /**
         * This indicates that multiple global or constant variables (across separate
         * CUDA source files in the application) share the same string name.
         */
        cudaErrorDuplicateVariableName = 43,

        /**
         * This indicates that multiple textures (across separate CUDA source
         * files in the application) share the same string name.
         */
        cudaErrorDuplicateTextureName = 44,

        /**
         * This indicates that multiple surfaces (across separate CUDA source
         * files in the application) share the same string name.
         */
        cudaErrorDuplicateSurfaceName = 45,

        /**
         * This indicates that all CUDA devices are busy or unavailable at the current
         * time. Devices are often busy/unavailable due to use of
         * ::cudaComputeModeExclusive, ::cudaComputeModeProhibited or when long
         * running CUDA kernels have filled up the GPU and are blocking new work
         * from starting. They can also be unavailable due to memory constraints
         * on a device that already has active CUDA work being performed.
         */
        cudaErrorDevicesUnavailable = 46,

        /**
         * This indicates that the device kernel image is invalid.
         */
        cudaErrorInvalidKernelImage = 47,

        /**
         * This indicates that there is no kernel image available that is suitable
         * for the device. This can occur when a user specifies code generation
         * options for a particular CUDA source file that do not include the
         * corresponding device configuration.
         */
        cudaErrorNoKernelImageForDevice = 48,

        /**
         * This indicates that the current context is not compatible with this
         * the CUDA Runtime. This can only occur if you are using CUDA
         * Runtime/Driver interoperability and have created an existing Driver
         * context using the driver API. The Driver context may be incompatible
         * either because the Driver context was created using an older version
         * of the API, because the Runtime API call expects a primary driver
         * context and the Driver context is not primary, or because the Driver
         * context has been destroyed. Please see \ref CUDART_DRIVER "Interactions
         * with the CUDA Driver API" for more information.
         */
        cudaErrorIncompatibleDriverContext = 49,

        /**
         * This error indicates that a call to ::cudaDeviceEnablePeerAccess() is
         * trying to re-enable peer addressing on from a context which has already
         * had peer addressing enabled.
         */
        cudaErrorPeerAccessAlreadyEnabled = 50,

        /**
         * This error indicates that ::cudaDeviceDisablePeerAccess() is trying to
         * disable peer addressing which has not been enabled yet via
         * ::cudaDeviceEnablePeerAccess().
         */
        cudaErrorPeerAccessNotEnabled = 51,

        /**
         * This indicates that a call tried to access an exclusive-thread device that
         * is already in use by a different thread.
         */
        cudaErrorDeviceAlreadyInUse = 54,

        /**
         * This indicates profiler is not initialized for this run. This can
         * happen when the application is running with external profiling tools
         * like visual profiler.
         */
        cudaErrorProfilerDisabled = 55,

        /**
         * \deprecated
         * This error return is deprecated as of CUDA 5.0. It is no longer an error
         * to attempt to enable/disable the profiling via ::cudaProfilerStart or
         * ::cudaProfilerStop without initialization.
         */
        cudaErrorProfilerNotInitialized = 56,

        /**
         * \deprecated
         * This error return is deprecated as of CUDA 5.0. It is no longer an error
         * to call cudaProfilerStart() when profiling is already enabled.
         */
        cudaErrorProfilerAlreadyStarted = 57,

        /**
         * \deprecated
         * This error return is deprecated as of CUDA 5.0. It is no longer an error
         * to call cudaProfilerStop() when profiling is already disabled.
         */
        cudaErrorProfilerAlreadyStopped = 58,

        /**
         * An assert triggered in device code during kernel execution. The device
         * cannot be used again until ::cudaThreadExit() is called. All existing
         * allocations are invalid and must be reconstructed if the program is to
         * continue using CUDA.
         */
        cudaErrorAssert = 59,

        /**
         * This error indicates that the hardware resources required to enable
         * peer access have been exhausted for one or more of the devices
         * passed to ::cudaEnablePeerAccess().
         */
        cudaErrorTooManyPeers = 60,

        /**
         * This error indicates that the memory range passed to ::cudaHostRegister()
         * has already been registered.
         */
        cudaErrorHostMemoryAlreadyRegistered = 61,

        /**
         * This error indicates that the pointer passed to ::cudaHostUnregister()
         * does not correspond to any currently registered memory region.
         */
        cudaErrorHostMemoryNotRegistered = 62,

        /**
         * This error indicates that an OS call failed.
         */
        cudaErrorOperatingSystem = 63,

        /**
         * This error indicates that P2P access is not supported across the given
         * devices.
         */
        cudaErrorPeerAccessUnsupported = 64,

        /**
         * This error indicates that a device runtime grid launch did not occur
         * because the depth of the child grid would exceed the maximum supported
         * number of nested grid launches.
         */
        cudaErrorLaunchMaxDepthExceeded = 65,

        /**
         * This error indicates that a grid launch did not occur because the kernel
         * uses file-scoped textures which are unsupported by the device runtime.
         * Kernels launched via the device runtime only support textures created with
         * the Texture Object API's.
         */
        cudaErrorLaunchFileScopedTex = 66,

        /**
         * This error indicates that a grid launch did not occur because the kernel
         * uses file-scoped surfaces which are unsupported by the device runtime.
         * Kernels launched via the device runtime only support surfaces created with
         * the Surface Object API's.
         */
        cudaErrorLaunchFileScopedSurf = 67,

        /**
         * This error indicates that a call to ::cudaDeviceSynchronize made from
         * the device runtime failed because the call was made at grid depth greater
         * than than either the default (2 levels of grids) or user specified device
         * limit ::cudaLimitDevRuntimeSyncDepth. To be able to synchronize on
         * launched grids at a greater depth successfully, the maximum nested
         * depth at which ::cudaDeviceSynchronize will be called must be specified
         * with the ::cudaLimitDevRuntimeSyncDepth limit to the ::cudaDeviceSetLimit
         * api before the host-side launch of a kernel using the device runtime.
         * Keep in mind that additional levels of sync depth require the runtime
         * to reserve large amounts of device memory that cannot be used for
         * user allocations.
         */
        cudaErrorSyncDepthExceeded = 68,

        /**
         * This error indicates that a device runtime grid launch failed because
         * the launch would exceed the limit ::cudaLimitDevRuntimePendingLaunchCount.
         * For this launch to proceed successfully, ::cudaDeviceSetLimit must be
         * called to set the ::cudaLimitDevRuntimePendingLaunchCount to be higher
         * than the upper bound of outstanding launches that can be issued to the
         * device runtime. Keep in mind that raising the limit of pending device
         * runtime launches will require the runtime to reserve device memory that
         * cannot be used for user allocations.
         */
        cudaErrorLaunchPendingCountExceeded = 69,

        /**
         * This error indicates the attempted operation is not permitted.
         */
        cudaErrorNotPermitted = 70,

        /**
         * This error indicates the attempted operation is not supported
         * on the current system or device.
         */
        cudaErrorNotSupported = 71,

        /**
         * Device encountered an error in the call stack during kernel execution,
         * possibly due to stack corruption or exceeding the stack size limit.
         * The context cannot be used, so it must be destroyed (and a new one should be created).
         * All existing device memory allocations from this context are invalid
         * and must be reconstructed if the program is to continue using CUDA.
         */
        cudaErrorHardwareStackError = 72,

        /**
         * The device encountered an illegal instruction during kernel execution
         * The context cannot be used, so it must be destroyed (and a new one should be created).
         * All existing device memory allocations from this context are invalid
         * and must be reconstructed if the program is to continue using CUDA.
         */
        cudaErrorIllegalInstruction = 73,

        /**
         * The device encountered a load or store instruction
         * on a memory address which is not aligned.
         * The context cannot be used, so it must be destroyed (and a new one should be created).
         * All existing device memory allocations from this context are invalid
         * and must be reconstructed if the program is to continue using CUDA.
         */
        cudaErrorMisalignedAddress = 74,

        /**
         * While executing a kernel, the device encountered an instruction
         * which can only operate on memory locations in certain address spaces
         * (global, shared, or local), but was supplied a memory address not
         * belonging to an allowed address space.
         * The context cannot be used, so it must be destroyed (and a new one should be created).
         * All existing device memory allocations from this context are invalid
         * and must be reconstructed if the program is to continue using CUDA.
         */
        cudaErrorInvalidAddressSpace = 75,

        /**
         * The device encountered an invalid program counter.
         * The context cannot be used, so it must be destroyed (and a new one should be created).
         * All existing device memory allocations from this context are invalid
         * and must be reconstructed if the program is to continue using CUDA.
         */
        cudaErrorInvalidPc = 76,

        /**
         * The device encountered a load or store instruction on an invalid memory address.
         * The context cannot be used, so it must be destroyed (and a new one should be created).
         * All existing device memory allocations from this context are invalid
         * and must be reconstructed if the program is to continue using CUDA.
         */
        cudaErrorIllegalAddress = 77,

        /**
         * A PTX compilation failed. The runtime may fall back to compiling PTX if
         * an application does not contain a suitable binary for the current device.
         */
        cudaErrorInvalidPtx = 78,

        /**
         * This indicates an error with the OpenGL or DirectX context.
         */
        cudaErrorInvalidGraphicsContext = 79,

        /**
         * This indicates an internal startup failure in the CUDA runtime.
         */
        cudaErrorStartupFailure = 0x7f,

        /**
         * Any unhandled CUDA driver error is added to this value and returned via
         * the runtime. Production releases of CUDA should not return such errors.
         * \deprecated
         * This error return is deprecated as of CUDA 4.1.
         */
        cudaErrorApiFailureBase = 10000
    }

    /// <summary>
    /// CUDA memory copy types
    /// </summary>
    public enum cudaMemcpyKind
    {
        HostToHost = 0,      // Host   -> Host
        HostToDevice = 1,    // Host   -> Device
        DeviceToHost = 2,    // Device -> Host
        DeviceToDevice = 3,  // Device -> Device
        Default = 4          // Default based unified virtual address space
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct CUuuid_st
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
        public sbyte[] bytes;
    }

    /// <summary>
    /// CUDA device properties
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cudaDeviceProp
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)]
        public sbyte[] name;                  // ASCII string identifying device
        public cudaUUID_t uuid;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
        public sbyte[] luid;
        public uint luidDeviceNodeMask;
        public Int64 totalGlobalMem;             // Global memory available on device in bytes
        public Int64 sharedMemPerBlock;          // Shared memory available per block in bytes
        public int regsPerBlock;               // 32-bit registers available per block
        public int warpSize;                   // Warp size in threads
        public Int64 memPitch;                   // Maximum pitch in bytes allowed by memory copies
        public int maxThreadsPerBlock;         // Maximum number of threads per block
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxThreadsDim;           // Maximum size of each dimension of a block
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxGridSize;             // Maximum size of each dimension of a grid
        public int clockRate;                  // Clock frequency in kilohertz
        public Int64 totalConstMem;              // Constant memory available on device in bytes
        public int major;                      // Major compute capability
        public int minor;                      // Minor compute capability
        public Int64 textureAlignment;           // Alignment requirement for textures
        public Int64 texturePitchAlignment;      // Pitch alignment requirement for texture references bound to pitched memory
        public int deviceOverlap;              // Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount.
        public int multiProcessorCount;        // Number of multiprocessors on device
        public int kernelExecTimeoutEnabled;   // Specified whether there is a run time limit on kernels
        public int integrated;                 // Device is integrated as opposed to discrete
        public int canMapHostMemory;           // Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer
        public int computeMode;                // Compute mode (See ::cudaComputeMode)
        public int maxTexture1D;               // Maximum 1D texture size
        public int maxTexture1DMipmap;         // Maximum 1D mipmapped texture size
        public int maxTexture1DLinear;         // Maximum size for 1D textures bound to linear memory
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxTexture2D;            // Maximum 2D texture dimensions
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxTexture2DMipmap;      // Maximum 2D mipmapped texture dimensions
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxTexture2DLinear;      // Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxTexture2DGather;      // Maximum 2D texture dimensions if texture gather operations have to be performed
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxTexture3D;            // Maximum 3D texture dimensions
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxTexture3DAlt;         // Maximum alternate 3D texture dimensions
        public int maxTextureCubemap;          // Maximum Cubemap texture dimensions
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxTexture1DLayered;     // Maximum 1D layered texture dimensions
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxTexture2DLayered;     // Maximum 2D layered texture dimensions
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxTextureCubemapLayered;// Maximum Cubemap layered texture dimensions
        public int maxSurface1D;               // Maximum 1D surface size
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxSurface2D;            // Maximum 2D surface dimensions
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxSurface3D;            // Maximum 3D surface dimensions
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxSurface1DLayered;     // Maximum 1D layered surface dimensions
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] maxSurface2DLayered;     // Maximum 2D layered surface dimensions
        public int maxSurfaceCubemap;          // Maximum Cubemap surface dimensions
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public int[] maxSurfaceCubemapLayered;// Maximum Cubemap layered surface dimensions
        public Int64 surfaceAlignment;           // Alignment requirements for surfaces
        public int concurrentKernels;          // Device can possibly execute multiple kernels concurrently
        public int ECCEnabled;                 // Device has ECC support enabled
        public int pciBusID;                   // PCI bus ID of the device
        public int pciDeviceID;                // PCI device ID of the device
        public int pciDomainID;                // PCI domain ID of the device
        public int tccDriver;                  // 1 if device is a Tesla device using TCC driver, 0 otherwise
        public int asyncEngineCount;           // Number of asynchronous engines
        public int unifiedAddressing;          // Device shares a unified address space with the host
        public int memoryClockRate;            // Peak memory clock frequency in kilohertz
        public int memoryBusWidth;             // Global memory bus width in bits
        public int l2CacheSize;                // Size of L2 cache in bytes
        public int maxThreadsPerMultiProcessor;// Maximum resident threads per multiprocessor
        public int streamPrioritiesSupported;  // Device supports stream priorities
        public int globalL1CacheSupported;     // Device supports caching globals in L1
        public int localL1CacheSupported;      // Device supports caching locals in L1
        public Int64 sharedMemPerMultiprocessor; // Shared memory available per multiprocessor in bytes
        public int regsPerMultiprocessor;      // 32-bit registers available per multiprocessor
        public int managedMemory;              // Device supports allocating managed memory on this system
        public int isMultiGpuBoard;            // Device is on a multi-GPU board
        public int multiGpuBoardGroupID;       // Unique identifier for a group of devices on the same multi-GPU board
        public int hostNativeAtomicSupported;  // Link between the device and the host supports native atomic operations
        public int singleToDoublePrecisionPerfRatio; // Ratio of single precision performance (in floating-point operations per second) to double precision performance
        public int pageableMemoryAccess;       // Device supports coherently accessing pageable memory without calling cudaHostRegister on it
        public int concurrentManagedAccess;    // Device can coherently access managed memory concurrently with the CPU
        public int computePreemptionSupported; // Device supports Compute Preemption
        public int canUseHostPointerForRegisteredMem; // Device can access host registered memory at the same virtual address as the CPU
        public int cooperativeLaunch;          // Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel
        public int cooperativeMultiDeviceLaunch; // Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated.
        public Int64 sharedMemPerBlockOptin;     // Per device maximum shared memory per block usable by special opt in
        public int pageableMemoryAccessUsesHostPageTables; // Device accesses pageable memory via the host's page tables
        public int directManagedMemAccessFromHost; // Host can directly access managed memory on the device without migration.
        public int maxBlocksPerMultiProcessor; // Maximum number of resident blocks per multiprocessor
        public int accessPolicyMaxWindowSize;  // The maximum value of ::cudaAccessPolicyWindow::num_bytes.
        public Int64 reservedSharedMemPerBlock;  // Shared memory reserved by CUDA driver per block in bytes
    }

    /// <summary>
    /// Channel format kind
    /// </summary>
    public enum cudaChannelFormatKind
    {
        cudaChannelFormatKindSigned = 0,      // Signed channel format
        cudaChannelFormatKindUnsigned = 1,      // Unsigned channel format
        cudaChannelFormatKindFloat = 2,      // Float channel format
        cudaChannelFormatKindNone = 3       // No channel format
    }

    /// <summary>
    /// CUDA Channel format descriptor
    /// </summary>
    public struct cudaChannelFormatDesc
    {
        public int x; // x
        public int y; // y
        public int z; // z
        public int w; // w
        public cudaChannelFormatKind f; // Channel format kind
    }

    /// <summary>
    /// CUDA extent
    /// </summary>
    public struct cudaExtent
    {
        public long width;     // Width in elements when referring to array memory, in bytes when referring to linear memory
        public long height;    // Height in elements
        public long depth;     // Depth in elements
    }
}