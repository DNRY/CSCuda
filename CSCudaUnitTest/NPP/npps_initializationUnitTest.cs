using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CSCuda;
using CSCuda.NPP;
using System.Runtime.InteropServices;

namespace CSCudaUnitTest.NPP
{
    [TestClass]
    public class npps_initializationUnitTest
    {
        [TestMethod]
        public void nppsSet_32s_test()
        {
            int length = 1024;
            int value = 75;
            IntPtr ptr = Npps.nppsMalloc_32s(length);

            int[] result = new int[length];

            GCHandle gcHandle = GCHandle.Alloc(result, GCHandleType.Pinned);
            IntPtr h_result = Marshal.UnsafeAddrOfPinnedArrayElement(result, 0);
            UInt64 size = Convert.ToUInt64(sizeof(int) * result.Length);

            NppStatus status = Npps.nppsSet_32s(value, ptr, length);
            if (status != NppStatus.NPP_SUCCESS)
            {
                Assert.Fail(String.Format("Fail {0}", status.ToString()));
            }

            cudaError cudaStatus = CudaRuntimeApi.cudaMemcpy(h_result, ptr, size, cudaMemcpyKind.DeviceToHost);
            if (cudaStatus != cudaError.cudaSuccess)
            {
                Assert.Fail(String.Format("Fail {0}", cudaStatus.ToString()));
            }

            for (int i = 0; i < result.Length; i++)
            {
                Assert.AreEqual(value, result[i]);
            }

            gcHandle.Free();
            Npps.nppsFree(ptr);
        }

        [TestMethod]
        public void nppsZero_32s_test()
        {
            NppStatus status;
            cudaError cudaStatus;
            int length = 1024;
            int value = 75;
            IntPtr ptr = Npps.nppsMalloc_32s(length);

            int[] result = new int[length];

            GCHandle gcHandle = GCHandle.Alloc(result, GCHandleType.Pinned);
            IntPtr h_result = Marshal.UnsafeAddrOfPinnedArrayElement(result, 0);
            UInt64 size = Convert.ToUInt64(sizeof(int) * result.Length);

            status = Npps.nppsSet_32s(value, ptr, length);
            if (status != NppStatus.NPP_SUCCESS)
            {
                Assert.Fail(String.Format("Fail {0}", status.ToString()));
            }

            cudaStatus = CudaRuntimeApi.cudaMemcpy(h_result, ptr, size, cudaMemcpyKind.DeviceToHost);
            if (cudaStatus != cudaError.cudaSuccess)
            {
                Assert.Fail(String.Format("Fail {0}", cudaStatus.ToString()));
            }

            for (int i = 0; i < result.Length; i++)
            {
                Assert.AreEqual(value, result[i]);
            }

            status = Npps.nppsZero_32s(ptr, length);
            cudaStatus = CudaRuntimeApi.cudaMemcpy(h_result, ptr, size, cudaMemcpyKind.DeviceToHost);
            if (cudaStatus != cudaError.cudaSuccess)
            {
                Assert.Fail(String.Format("Fail {0}", cudaStatus.ToString()));
            }

            for (int i = 0; i < result.Length; i++)
            {
                Assert.AreEqual(0, result[i]);
            }

            gcHandle.Free();
            Npps.nppsFree(ptr);
        }

        [TestMethod]
        public void nppsCopy_32s_test()
        {
            NppStatus status;
            cudaError cudaStatus;
            int length = 1024;
            int value = 75;
            IntPtr d_src = Npps.nppsMalloc_32s(length);
            IntPtr d_dst = Npps.nppsMalloc_32s(length);

            int[] result = new int[length];

            GCHandle gcHandle = GCHandle.Alloc(result, GCHandleType.Pinned);
            IntPtr h_result = Marshal.UnsafeAddrOfPinnedArrayElement(result, 0);
            UInt64 size = Convert.ToUInt64(sizeof(int) * result.Length);

            status = Npps.nppsSet_32s(value, d_src, length);
            if (status != NppStatus.NPP_SUCCESS)
            {
                Assert.Fail(String.Format("Fail {0}", status.ToString()));
            }

            status = Npps.nppsCopy_32s(d_src, d_dst, length);
            if (status != NppStatus.NPP_SUCCESS)
            {
                Assert.Fail(String.Format("Fail {0}", status.ToString()));
            }

            cudaStatus = CudaRuntimeApi.cudaMemcpy(h_result, d_dst, size, cudaMemcpyKind.DeviceToHost);
            if (cudaStatus != cudaError.cudaSuccess)
            {
                Assert.Fail(String.Format("Fail {0}", cudaStatus.ToString()));
            }

            for (int i = 0; i < result.Length; i++)
            {
                Assert.AreEqual(value, result[i]);
            }

            gcHandle.Free();
            Npps.nppsFree(d_src);
        }
    }
}
