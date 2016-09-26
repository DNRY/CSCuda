using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace CSCuda
{
    public static class CudaRuntimeApi
    {
        private const string cudartString = @"cudart64_80";

        [DllImport(cudartString, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError cudaDeviceReset();

        [DllImport(cudartString, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError cudaDeviceSynchronize();

        [DllImport(cudartString, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError cudaGetDeviceCount(
            ref int count
            );

        [DllImport(cudartString, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError cudaGetDeviceProperties(
            ref cudaDeviceProp prop,
            int device
            );

        [DllImport(cudartString, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError cudaSetDevice(
            int device
            );

        [DllImport(cudartString, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError cudaStreamCreate(
            ref IntPtr stream
            );

        [DllImport(cudartString, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError cudaStreamDestroy(
            IntPtr stream
            );

        [DllImport(cudartString, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError cudaMalloc(
            ref IntPtr devPtr,
            ulong size
            );

        [DllImport(cudartString, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError cudaMallocArray(
            ref IntPtr array,
            ref cudaChannelFormatDesc desc,
            ulong width,
            ulong height = 0,
            uint flags = 0
            );

        [DllImport(cudartString, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError cudaFree(
            IntPtr devPtr
            );

        [DllImport(cudartString, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError cudaFreeHost(
            IntPtr ptr
            );

        [DllImport(cudartString, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError cudaHostAlloc(
            ref IntPtr devPtr,
            ulong size,
            uint flags
            );

        [DllImport(cudartString, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError cudaMalloc3DArray(
            ref IntPtr array,
            ref cudaChannelFormatDesc desc,
            cudaExtent extent,
            uint flags = 0);

        [DllImport(cudartString, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError cudaMemGetInfo(
            ref ulong free,
            ref ulong total
            );

        [DllImport(cudartString, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError cudaMemcpy(
            IntPtr dst,
            IntPtr src,
            ulong count,
            cudaMemcpyKind kind
            );

        [DllImport(cudartString, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError cudaMemcpy2D(
            IntPtr dst,
            ulong dpitch,
            IntPtr src,
            ulong spitch,
            ulong width,
            ulong height,
            int cudaMemcpyKind
            );

        [DllImport(cudartString, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError cudaMemset(
            IntPtr devPtr,
            int value,
            ulong count
            );

        [DllImport(cudartString, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError cudaDriverGetVersion(
            ref int driverVersion
            );

        [DllImport(cudartString, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        public static extern cudaError cudaRuntimeGetVersion(
            ref int runtimeVersion
            );
    }
}