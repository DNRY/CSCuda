using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CSCuda.NPP;
using System.Runtime.InteropServices;
using CSCuda;
using System.Linq;

namespace CSCudaUnitTest.NPP
{
    [TestClass]
    public class npps_conversion_functionsUnitTest
    {
        [TestMethod]
        public void nppsConvert_32s32f_test()
        {
            NppStatus nppStatus;
            cudaError cudaStatus;
            int length = 1024;
            int value = 75;
            float expected = 75.0F;

            IntPtr d_src = Npps.nppsMalloc_32s(length);
            IntPtr d_dst = Npps.nppsMalloc_32f(length);

            float[] result = new float[length];

            GCHandle gcHandle = GCHandle.Alloc(result, GCHandleType.Pinned);
            IntPtr h_result = Marshal.UnsafeAddrOfPinnedArrayElement(result, 0);
            UInt64 size = Convert.ToUInt64(sizeof(int) * result.Length);

            nppStatus = Npps.nppsSet_32s(value, d_src, length);
            if (nppStatus != NppStatus.NPP_SUCCESS)
            {
                Assert.Fail(String.Format("Fail {0}", nppStatus.ToString()));
            }

            nppStatus = Npps.nppsConvert_32s32f(d_src, d_dst, length);
            if (nppStatus != NppStatus.NPP_SUCCESS)
            {
                Assert.Fail(String.Format("Fail {0}", nppStatus.ToString()));
            }


            cudaStatus = CudaRuntimeApi.cudaMemcpy(h_result, d_dst, size, cudaMemcpyKind.DeviceToHost);
            if (cudaStatus != cudaError.cudaSuccess)
            {
                Assert.Fail(String.Format("Fail {0}", cudaStatus.ToString()));
            }

            for (int i = 0; i < result.Length; i++)
            {
                Assert.AreEqual(expected, result[i]);
            }

            gcHandle.Free();
            Npps.nppsFree(d_src);
            Npps.nppsFree(d_dst);
        }

        [TestMethod]
        public void nppsConvert_32f16s_test()
        {
            NppStatus nppStatus;
            cudaError cudaStatus;
            int length = 1024;
            int value = 75;
            float expected = 75.0F;

            IntPtr d_src = Npps.nppsMalloc_32f(length);
            IntPtr d_dst = Npps.nppsMalloc_16s(length);

            short[] result = new short[length];

            GCHandle gcHandle = GCHandle.Alloc(result, GCHandleType.Pinned);
            IntPtr h_result = Marshal.UnsafeAddrOfPinnedArrayElement(result, 0);
            UInt64 size = Convert.ToUInt64(sizeof(short) * result.Length);

            nppStatus = Npps.nppsSet_32f((float)value, d_src, length);
            if (nppStatus != NppStatus.NPP_SUCCESS)
            {
                Assert.Fail(String.Format("Fail {0}", nppStatus.ToString()));
            }

            nppStatus = Npps.nppsConvert_32f16s_Sfs(d_src, d_dst, length, NppRoundMode.NPP_RND_NEAR, 0);
            if (nppStatus != NppStatus.NPP_SUCCESS)
            {
                Assert.Fail(String.Format("Fail {0}", nppStatus.ToString()));
            }


            cudaStatus = CudaRuntimeApi.cudaMemcpy(h_result, d_dst, size, cudaMemcpyKind.DeviceToHost);
            if (cudaStatus != cudaError.cudaSuccess)
            {
                Assert.Fail(String.Format("Fail {0}", cudaStatus.ToString()));
            }

            for (int i = 0; i < result.Length; i++)
            {
                Assert.AreEqual(expected, result[i]);
            }

            gcHandle.Free();
            Npps.nppsFree(d_src);
            Npps.nppsFree(d_dst);
        }

        [TestMethod]
        public void nppsThreshold_32f_test()
        {
            NppStatus nppStatus;
            cudaError cudaStatus;
            int width = 128;
            int height = 24; 
            int length = width * height;

            float level = 10.0F;

            IntPtr d_src = Npps.nppsMalloc_32s(length);
            IntPtr d_dst = Npps.nppsMalloc_32f(length);

            float[] input = new float[length];
            float[] result = new float[length];

            float[] line = Array.ConvertAll(Enumerable.Range(0, width).ToArray(), Convert.ToSingle);
            for (int i = 0; i < height; i++)
            {
                Array.Copy(line, 0, input, i * width, width);
            }

            UInt64 size = Convert.ToUInt64(sizeof(int) * result.Length);

            GCHandle gchInput = GCHandle.Alloc(input, GCHandleType.Pinned);
            IntPtr h_input = Marshal.UnsafeAddrOfPinnedArrayElement(input, 0);
            cudaStatus = CudaRuntimeApi.cudaMemcpy(d_src, h_input, size, cudaMemcpyKind.HostToDevice);
            if (cudaStatus != cudaError.cudaSuccess)
            {
                Assert.Fail(String.Format("Fail {0}", cudaStatus.ToString()));
            }

            nppStatus = Npps.nppsThreshold_32f(d_src, d_dst, length, level, NppCmpOp.NPP_CMP_LESS);
            if (nppStatus != NppStatus.NPP_SUCCESS)
            {
                Assert.Fail(String.Format("Fail {0}", nppStatus.ToString()));
            }

            GCHandle gchResult = GCHandle.Alloc(result, GCHandleType.Pinned);
            IntPtr h_result = Marshal.UnsafeAddrOfPinnedArrayElement(result, 0);            
            cudaStatus = CudaRuntimeApi.cudaMemcpy(h_result, d_dst, size, cudaMemcpyKind.DeviceToHost);
            if (cudaStatus != cudaError.cudaSuccess)
            {
                Assert.Fail(String.Format("Fail {0}", cudaStatus.ToString()));
            }

            for (int i = 0; i < result.Length; i++)
            {
                if (result[i] < level)
                    Assert.Fail(String.Format("Fail. level : {0}, value :{1}", level, result[i]));
            }

            gchInput.Free();
            gchResult.Free();
            
            Npps.nppsFree(d_src);
            Npps.nppsFree(d_dst);
        }
    }
}
