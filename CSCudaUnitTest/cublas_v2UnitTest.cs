using CSCuda;
using CSCuda.CUBLAS;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Runtime.InteropServices;

namespace CSCudaUnitTest
{
    [TestClass]
    public class cublas_v2UnitTest
    {
        [TestMethod]
        public void cublasCreate_cublasDestroy_test()
        {
            var handle = IntPtr.Zero;
            var cblasStatus = Cublas_v2.cublasCreate(ref handle);
            Assert.AreEqual(cublasStatus_t.CUBLAS_STATUS_SUCCESS, cblasStatus);

            cblasStatus = Cublas_v2.cublasDestroy(handle);
            Assert.AreEqual(cublasStatus_t.CUBLAS_STATUS_SUCCESS, cblasStatus);
        }

        [TestMethod]
        public void cublasSgemm_test()
        {
            int devCount = 0;
            var status = CudaRuntimeApi.cudaGetDeviceCount(ref devCount);
            Assert.AreEqual(cudaError.cudaSuccess, status);

            int devId = 0;
            status = CudaRuntimeApi.cudaSetDevice(devId);

            var handle = IntPtr.Zero;
            var cblasStatus = Cublas_v2.cublasCreate(ref handle);

            Random rand = new Random();
            int rows_a = rand.Next(2, 10);
            int cols_a = rand.Next(2, 10);

            int rows_b = cols_a;
            int cols_b = rand.Next(2, 10);

            int rows_c = rows_a;
            int cols_c = cols_b;

            float alpha = 1.0F;
            float beta = 0.0F;

            float[] A = new float[rows_a * cols_a];
            float[] B = new float[rows_b * cols_b];
            float[] C = new float[rows_c * cols_c];
            float[] resultC = new float[rows_c * cols_c];

            for (int i = 0; i < A.Length; i++)
            {
                A[i] = Convert.ToSingle(rand.Next(0, 10));
            }

            for (int i = 0; i < B.Length; i++)
            {
                B[i] = Convert.ToSingle(rand.Next(0, 10));
            }

            var d_a = IntPtr.Zero;
            var d_b = IntPtr.Zero;
            var d_c = IntPtr.Zero;

            status = CudaRuntimeApi.cudaMalloc(ref d_a, (ulong)A.Length * sizeof(float));
            status = CudaRuntimeApi.cudaMalloc(ref d_b, (ulong)B.Length * sizeof(float));
            status = CudaRuntimeApi.cudaMalloc(ref d_c, (ulong)C.Length * sizeof(float));

            var gch_a = GCHandle.Alloc(A, GCHandleType.Pinned);
            var gch_b = GCHandle.Alloc(B, GCHandleType.Pinned);
            var gch_c = GCHandle.Alloc(C, GCHandleType.Pinned);
            var gch_resultC = GCHandle.Alloc(resultC, GCHandleType.Pinned);

            var h_a = Marshal.UnsafeAddrOfPinnedArrayElement(A, 0);
            var h_b = Marshal.UnsafeAddrOfPinnedArrayElement(B, 0);
            var h_c = Marshal.UnsafeAddrOfPinnedArrayElement(C, 0);
            var h_resultC = Marshal.UnsafeAddrOfPinnedArrayElement(resultC, 0);

            status = CudaRuntimeApi.cudaMemcpy(d_a, h_a, (ulong)A.Length * sizeof(float), cudaMemcpyKind.HostToDevice);
            status = CudaRuntimeApi.cudaMemcpy(d_b, h_b, (ulong)B.Length * sizeof(float), cudaMemcpyKind.HostToDevice);
            status = CudaRuntimeApi.cudaMemcpy(d_c, h_c, (ulong)C.Length * sizeof(float), cudaMemcpyKind.HostToDevice);

            cblasStatus = Cublas_v2.cublasSgemm(
                        handle,
                        cublasOperation_t.CUBLAS_OP_N,
                        cublasOperation_t.CUBLAS_OP_N,
                        rows_a,
                        cols_b,
                        cols_a,
                        ref alpha,
                        d_a,
                        rows_a,
                        d_b,
                        rows_b,
                        ref beta,
                        d_c,
                        rows_c
                        );

            status = CudaRuntimeApi.cudaMemcpy(h_resultC, d_c, (ulong)C.Length * sizeof(float), cudaMemcpyKind.DeviceToHost);
            var mResultC = Matrix<float>.Build.Dense(rows_c, cols_c, resultC);
            var mA = Matrix<float>.Build.Dense(rows_a, cols_a, A);
            var mB = Matrix<float>.Build.Dense(rows_b, cols_b, B);
            var mExpectedC = Matrix<float>.Build.Dense(rows_c, cols_c, C).Clone();
            mExpectedC = alpha * mA * mB + beta * mExpectedC;
            float[] expected = mExpectedC.ToColumnWiseArray();

            Console.WriteLine("alpha : {0}, beta : {1}", alpha, beta);
            Console.WriteLine("A");
            Console.WriteLine(mA.ToString());
            Console.WriteLine();
            Console.WriteLine("B");
            Console.WriteLine(mB.ToString());
            Console.WriteLine();
            Console.WriteLine("resultC");
            Console.WriteLine(mResultC.ToString());
            Console.WriteLine();
            Console.WriteLine("expectedC");
            Console.WriteLine(mExpectedC.ToString());

            for (int i = 0; i < C.Length; i++)
            {
                Assert.AreEqual(expected[i], resultC[i]);
            }

            cblasStatus = Cublas_v2.cublasDestroy(handle);

            status = CudaRuntimeApi.cudaFree(d_a);
            status = CudaRuntimeApi.cudaFree(d_b);
            status = CudaRuntimeApi.cudaFree(d_c);

            gch_a.Free();
            gch_b.Free();
            gch_c.Free();
        }

        [TestMethod]
        public void cublasCgemm_test()
        {
            int devCount = 0;
            var status = CudaRuntimeApi.cudaGetDeviceCount(ref devCount);
            Assert.AreEqual(cudaError.cudaSuccess, status);

            int devId = 0;
            status = CudaRuntimeApi.cudaSetDevice(devId);

            var handle = IntPtr.Zero;
            var cblasStatus = Cublas_v2.cublasCreate(ref handle);

            Random rand = new Random();
            int rows_a = rand.Next(2, 10);
            int cols_a = rand.Next(2, 10);

            int rows_b = cols_a;
            int cols_b = rand.Next(2, 10);

            int rows_c = rows_a;
            int cols_c = cols_b;
            var A = new float2[rows_a * cols_a];
            var B = new float2[rows_b * cols_b];
            var C = new float2[rows_c * cols_c];
            var resultC = new float2[rows_c * cols_c];

            var cA = new Complex32[rows_a * cols_a];
            var cB = new Complex32[rows_b * cols_b];
            var cC = new Complex32[rows_c * cols_c];
            var cResultC = new Complex32[rows_c * cols_c];

            for (int i = 0; i < A.Length; i++)
            {
                var real = Convert.ToSingle(rand.Next(0, 10));
                var imag = Convert.ToSingle(rand.Next(0, 10));
                A[i] = new float2(real, imag);
                cA[i] = new Complex32(real, imag);
            }

            for (int i = 0; i < B.Length; i++)
            {
                var real = Convert.ToSingle(rand.Next(0, 10));
                var imag = Convert.ToSingle(rand.Next(0, 10));
                B[i] = new float2(real, imag);
                cB[i] = new Complex32(real, imag);
            }

            for (int i = 0; i < C.Length; i++)
            {
                var real = Convert.ToSingle(rand.Next(0, 10));
                var imag = Convert.ToSingle(rand.Next(0, 10));
                C[i] = new float2(real, imag);
                cC[i] = new Complex32(real, imag);
            }

            var alphaReal = Convert.ToSingle(rand.Next(0, 10));
            var alphaImag = Convert.ToSingle(rand.Next(0, 10));
            var alpha = new float2(alphaReal, alphaImag);
            var cAlpha = new Complex32(alphaReal, alphaImag);

            var betaReal = Convert.ToSingle(rand.Next(0, 10));
            var betaImag = Convert.ToSingle(rand.Next(0, 10));
            var beta = new float2(betaReal, betaImag);
            var cBeta = new Complex32(betaReal, betaImag);

            var d_a = IntPtr.Zero;
            var d_b = IntPtr.Zero;
            var d_c = IntPtr.Zero;

            status = CudaRuntimeApi.cudaMalloc(ref d_a, (ulong)(A.Length * Marshal.SizeOf(typeof(float2))));
            status = CudaRuntimeApi.cudaMalloc(ref d_b, (ulong)(B.Length * Marshal.SizeOf(typeof(float2))));
            status = CudaRuntimeApi.cudaMalloc(ref d_c, (ulong)(C.Length * Marshal.SizeOf(typeof(float2))));

            var gch_a = GCHandle.Alloc(A, GCHandleType.Pinned);
            var gch_b = GCHandle.Alloc(B, GCHandleType.Pinned);
            var gch_c = GCHandle.Alloc(C, GCHandleType.Pinned);
            var gch_resultC = GCHandle.Alloc(resultC, GCHandleType.Pinned);

            var h_a = Marshal.UnsafeAddrOfPinnedArrayElement(A, 0);
            var h_b = Marshal.UnsafeAddrOfPinnedArrayElement(B, 0);
            var h_c = Marshal.UnsafeAddrOfPinnedArrayElement(C, 0);
            var h_resultC = Marshal.UnsafeAddrOfPinnedArrayElement(resultC, 0);

            status = CudaRuntimeApi.cudaMemcpy(d_a, h_a, (ulong)(A.Length * Marshal.SizeOf(typeof(float2))), cudaMemcpyKind.HostToDevice);
            status = CudaRuntimeApi.cudaMemcpy(d_b, h_b, (ulong)(B.Length * Marshal.SizeOf(typeof(float2))), cudaMemcpyKind.HostToDevice);
            status = CudaRuntimeApi.cudaMemcpy(d_c, h_c, (ulong)(C.Length * Marshal.SizeOf(typeof(float2))), cudaMemcpyKind.HostToDevice);

            cblasStatus = Cublas_v2.cublasCgemm(
                        handle,
                        cublasOperation_t.CUBLAS_OP_N,
                        cublasOperation_t.CUBLAS_OP_N,
                        rows_a,
                        cols_b,
                        cols_a,
                        ref alpha,
                        d_a,
                        rows_a,
                        d_b,
                        rows_b,
                        ref beta,
                        d_c,
                        rows_c
                        );

            status = CudaRuntimeApi.cudaMemcpy(h_resultC, d_c, (ulong)(resultC.Length * Marshal.SizeOf(typeof(float2))), cudaMemcpyKind.DeviceToHost);
            for (int i = 0; i < rows_c * cols_c; i++)
            {
                cResultC[i] = new Complex32(resultC[i].X, resultC[i].Y);
            }
            var mResultC = Matrix<Complex32>.Build.Dense(rows_c, cols_c, cResultC);

            var mA = Matrix<Complex32>.Build.Dense(rows_a, cols_a, cA);
            var mB = Matrix<Complex32>.Build.Dense(rows_b, cols_b, cB);
            var mExpectedC = Matrix<Complex32>.Build.Dense(rows_c, cols_c, cC).Clone();
            mExpectedC = cAlpha * mA * mB + cBeta * mExpectedC;
            Complex32[] expected = mExpectedC.ToColumnWiseArray();

            Console.WriteLine("alpha : {0}, beta : {1}", alpha, beta);
            Console.WriteLine("A");
            Console.WriteLine(mA.ToString());
            Console.WriteLine();
            Console.WriteLine("B");
            Console.WriteLine(mB.ToString());
            Console.WriteLine();
            Console.WriteLine("resultC");
            Console.WriteLine(mResultC.ToString());
            Console.WriteLine();
            Console.WriteLine("expectedC");
            Console.WriteLine(mExpectedC.ToString());

            for (int i = 0; i < C.Length; i++)
            {
                Assert.AreEqual(expected[i], cResultC[i]);
            }

            cblasStatus = Cublas_v2.cublasDestroy(handle);

            status = CudaRuntimeApi.cudaFree(d_a);
            status = CudaRuntimeApi.cudaFree(d_b);
            status = CudaRuntimeApi.cudaFree(d_c);

            gch_a.Free();
            gch_b.Free();
            gch_c.Free();
        }
    }
}