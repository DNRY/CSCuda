using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

using CSCuda;
using cuComplex = CSCuda.float2;


namespace CSCuda.CUBLAS
{
    public enum cublasStatus_t
    {
        CUBLAS_STATUS_SUCCESS = 0,
        CUBLAS_STATUS_NOT_INITIALIZED = 1,
        CUBLAS_STATUS_ALLOC_FAILED = 3,
        CUBLAS_STATUS_INVALID_VALUE = 7,
        CUBLAS_STATUS_ARCH_MISMATCH = 8,
        CUBLAS_STATUS_MAPPING_ERROR = 11,
        CUBLAS_STATUS_EXECUTION_FAILED = 13,
        CUBLAS_STATUS_INTERNAL_ERROR = 14,
        CUBLAS_STATUS_NOT_SUPPORTED = 15,
        CUBLAS_STATUS_LICENSE_ERROR = 16
    }

    public enum cublasOperation_t
    {
        CUBLAS_OP_N = 0,
        CUBLAS_OP_T = 1,
        CUBLAS_OP_C = 2
    }

    public static class Cublas_api
    {
        internal const string dllName = @"cublas" + Constants.platform + "_" + Constants.major_version;

        [DllImport(dllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern cublasStatus_t cublasCreate_v2(ref IntPtr handle);

        [DllImport(dllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern cublasStatus_t cublasDestroy_v2(IntPtr handle);

        #region CUBLAS BLAS3 functions

        [DllImport(dllName, CallingConvention = CallingConvention.Cdecl)]
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

        [DllImport(dllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern cublasStatus_t cublasDgemm_v2(
            IntPtr handle,
            cublasOperation_t transa,
            cublasOperation_t transb,
            int m,
            int n,
            int k,
            ref double alpha,
            IntPtr A,
            int lda,
            IntPtr B,
            int ldb,
            ref double beta,
            IntPtr C,
            int ldc
            );

        [DllImport(dllName, SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
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
