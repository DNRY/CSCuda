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
        private const string cudartString = @"cudart64_75";

        [DllImport(cudartString, SetLastError = true)]
        public static extern cudaError cudaMalloc(
            ref IntPtr devPtr, 
            UInt64 size
            );

        [DllImport(cudartString, SetLastError = true)]
        public static extern cudaError cudaFree(
            IntPtr devPtr
            );

        [DllImport(cudartString, SetLastError = true)]
        public static extern cudaError cudaMemcpy(
            IntPtr dst,
            IntPtr src,
            ulong count,
            cudaMemcpyKind kind
            );
    }
}
