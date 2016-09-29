using System;
using System.Runtime.InteropServices;
using cuComplex = CSCuda.float2;

namespace CSCuda.CUBLAS
{
    public static class Cublas_v2
    {
        [DllImport(Cublas_api.dllName, EntryPoint = "cublasCreate_v2", CallingConvention = CallingConvention.Cdecl)]
        public static extern cublasStatus_t cublasCreate(ref IntPtr handle);

        [DllImport(Cublas_api.dllName, EntryPoint = "cublasDestroy_v2", CallingConvention = CallingConvention.Cdecl)]
        public static extern cublasStatus_t cublasDestroy(IntPtr handle);

        [DllImport(Cublas_api.dllName, EntryPoint = "cublasSgemm_v2", CallingConvention = CallingConvention.Cdecl)]
        public static extern cublasStatus_t cublasSgemm(
            IntPtr handle,
            cublasOperation_t transa,
            cublasOperation_t transb,
            int m,
            int n,
            int k,
            ref float alpha,
            IntPtr A,
            int lda,
            IntPtr B,
            int ldb,
            ref float beta,
            IntPtr C,
            int ldc
            );

        [DllImport(Cublas_api.dllName, EntryPoint = "cublasCgemm_v2", CallingConvention = CallingConvention.Cdecl)]
        public static extern cublasStatus_t cublasCgemm(
            IntPtr handle,
            cublasOperation_t transa,
            cublasOperation_t transb,
            int m,
            int n,
            int k,
            ref cuComplex alpha,
            IntPtr A,
            int lda,
            IntPtr B,
            int ldb,
            ref cuComplex beta,
            IntPtr C,
            int ldc
            );
    }
}