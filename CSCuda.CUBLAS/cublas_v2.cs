using System;
using System.Runtime.InteropServices;
using cuComplex = CSCuda.float2;

namespace CSCuda.CUBLAS
{
    public static class Cublas_v2
    {
        private const string dllNamne = @"cublas64_80.dll";

        [DllImport(dllNamne, EntryPoint = "cublasCreate_v2", CallingConvention = CallingConvention.Cdecl)]
        public static extern cublasStatus_t cublasCreate(ref IntPtr handle);

        [DllImport(dllNamne, CallingConvention = CallingConvention.Cdecl)]
        public static extern cublasStatus_t cublasCreate_v2(ref IntPtr handle);

        [DllImport(dllNamne, EntryPoint = "cublasDestroy_v2", CallingConvention = CallingConvention.Cdecl)]
        public static extern cublasStatus_t cublasDestroy(IntPtr handle);

        [DllImport(dllNamne, CallingConvention = CallingConvention.Cdecl)]
        public static extern cublasStatus_t cublasDestroy_v2(IntPtr handle);

        #region CUBLAS BLAS3 functions

        [DllImport(dllNamne, EntryPoint = "cublasSgemm_v2", CallingConvention = CallingConvention.Cdecl)]
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

        [DllImport(dllNamne, CallingConvention = CallingConvention.Cdecl)]
        public static extern cublasStatus_t cublasSgemm_v2(
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

        [DllImport(dllNamne, EntryPoint = "cublasCgemm_v2", CallingConvention = CallingConvention.Cdecl)]
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

        [DllImport(dllNamne, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        public static extern cublasStatus_t cublasCgemm_v2(
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

        #endregion CUBLAS BLAS3 functions
    }
}