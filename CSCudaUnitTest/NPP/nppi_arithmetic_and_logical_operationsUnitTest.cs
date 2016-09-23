using CSCuda;
using CSCuda.NPP;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Linq;
using System.Runtime.InteropServices;

namespace CSCudaUnitTest.NPP
{
    [TestClass]
    public class nppi_arithmetic_and_logical_operationsUnitTest
    {
        [TestMethod]
        public void nppiAddC_32f_C3R_test()
        {
            NppStatus nppStatus;
            cudaError cudaStatus;
            int width = 256;
            int height = 256;
            int channel = 3;
            int stepInBytes = width * channel * sizeof(float);

            IntPtr d_src = Nppi.nppiMalloc_32f_C3(width, height, ref stepInBytes);
            IntPtr d_dst = Nppi.nppiMalloc_32f_C3(width, height, ref stepInBytes);
            
            float[] input = new float[width * height * channel];
            float[] result = new float[width * height * channel];
            float[] aconstant = Array.ConvertAll(Enumerable.Range(0, 3).ToArray(), Convert.ToSingle);
            
            UInt64 size = Convert.ToUInt64(sizeof(float) * result.Length);
            GCHandle gchInput = GCHandle.Alloc(input, GCHandleType.Pinned);
            IntPtr h_input = Marshal.UnsafeAddrOfPinnedArrayElement(input, 0);
            cudaStatus = CudaRuntimeApi.cudaMemcpy(d_src, h_input, size, cudaMemcpyKind.HostToDevice);
            if (cudaStatus != cudaError.cudaSuccess)
            {
                Assert.Fail(cudaStatus.ToString());
            }

            NppiSize roi;
            roi.width = width;
            roi.height = width;

            nppStatus = Nppi.nppiAddC_32f_C3R(d_src, width * channel * sizeof(float), aconstant, d_dst, width * channel * sizeof(float), roi);
            if (nppStatus != NppStatus.NPP_SUCCESS)
            {
                Assert.Fail(nppStatus.ToString());
            }

            GCHandle gchResult = GCHandle.Alloc(result, GCHandleType.Pinned);
            IntPtr h_result = Marshal.UnsafeAddrOfPinnedArrayElement(result, 0);
            cudaStatus = CudaRuntimeApi.cudaMemcpy(h_result, d_dst, size, cudaMemcpyKind.DeviceToHost);
            if (cudaStatus != cudaError.cudaSuccess)
            {
                Assert.Fail(cudaStatus.ToString());
            }

            for (int i = 0; i < result.Length; i++)
            {
                int aId = i % aconstant.Length;
                Assert.AreEqual(aconstant[aId], result[i]);
            }

            gchInput.Free();
            gchResult.Free();

            Nppi.nppiFree(d_src);
            Nppi.nppiFree(d_dst);
        }
    }
}