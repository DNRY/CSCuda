using CSCuda;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Runtime.InteropServices;
using System.Text;

namespace CSCudaUnitTest
{
    [TestClass]
    public class cuda_runtime_apiUnitTest
    {
        [TestMethod]
        public void cudaDeviceReset_test()
        {
            var status = CudaRuntimeApi.cudaDeviceReset();
            Assert.AreEqual(status, cudaError.cudaSuccess);
        }

        [TestMethod]
        public void cudaDeviceSynchronize_test()
        {
            var status = CudaRuntimeApi.cudaDeviceSynchronize();
            Assert.AreEqual(status, cudaError.cudaSuccess);
        }

        [TestMethod]
        public void cudaGetDeviceCount_test()
        {
            int count = 0;
            var status = CudaRuntimeApi.cudaGetDeviceCount(ref count);
            Assert.AreEqual(status, cudaError.cudaSuccess);
            Console.WriteLine("cuda device count : {0}", count);
        }

        public string HexStringFromByteArray(byte[] bytes)
        {
            var sb = new StringBuilder(bytes.Length * 2);
            foreach (var b in bytes)
            {
                sb.AppendFormat("{0:x2}", b);
            }
            return sb.ToString();
        }

        public byte[] UnsignedBytesFromSignedBytes(sbyte[] signed)
        {
            var unsigned = new byte[signed.Length];
            Buffer.BlockCopy(signed, 0, unsigned, 0, signed.Length);
            return unsigned;
        }

        [TestMethod]
        public void cudaSetDevice_cudaGetDeviceProperties_cudaDriverGetVersion_cudaRuntimeGetVersion_test()
        {
            int deviceCount = 0;
            var status = CudaRuntimeApi.cudaGetDeviceCount(ref deviceCount);
            Assert.AreEqual(status, cudaError.cudaSuccess);

            // This function call returns 0 if there are no CUDA capable devices.
            if (deviceCount == 0)
            {
                Console.WriteLine("There are no available device(s) that support CUDA");
            }
            else
            {
                Console.WriteLine("Detected {0} CUDA Capable device(s)", deviceCount);
            }

            for (int i = 0; i < deviceCount; i++)
            {
                status = CudaRuntimeApi.cudaSetDevice(i);
                Assert.AreEqual(status, cudaError.cudaSuccess);
                int driverVersion = 0;
                int runtimeVersion = 0;
                status = CudaRuntimeApi.cudaDriverGetVersion(ref driverVersion);
                Assert.AreEqual(status, cudaError.cudaSuccess);
                status = CudaRuntimeApi.cudaRuntimeGetVersion(ref runtimeVersion);
                Assert.AreEqual(status, cudaError.cudaSuccess);

                cudaDeviceProp deviceProp = new cudaDeviceProp();
                CudaRuntimeApi.cudaGetDeviceProperties(ref deviceProp, i);

                var uuid = HexStringFromByteArray(UnsignedBytesFromSignedBytes(deviceProp.uuid.bytes));
                unsafe
                {
                    fixed (sbyte* pName = deviceProp.name)
                    {
                        var name = new string(pName);
                        //Console.WriteLine("{0}, uuid = {1}", name, uuid);

                        Console.WriteLine("\nDevice {0}: \"{1}\", uuid = {2}", i, name, uuid);
                    }
                }

                Console.WriteLine("  CUDA Driver Version / Runtime Version          {0}.{1} / {2}.{3}", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
                Console.WriteLine("  CUDA Capability Major/Minor version number:    {0}.{1}", deviceProp.major, deviceProp.minor);
                Console.WriteLine(
                  "  Total amount of global memory:                 {0:0} MBytes ({1} bytes)",
                  Convert.ToSingle(deviceProp.totalGlobalMem / 1048576.0f),
                  deviceProp.totalGlobalMem);
            }
        }

        [TestMethod]
        public void cudaHostAlloc_cudaFreeHost_test()
        {
            IntPtr ptr = IntPtr.Zero;
            var length = 1024 * 1024;
            var size = 1024 * 1024 * sizeof(float);
            var status = CudaRuntimeApi.cudaHostAlloc(ref ptr, (ulong)size, 0);
            Assert.AreEqual(status, cudaError.cudaSuccess);
            Console.WriteLine($"ptr : {ptr}");

            float[] test = new float[length];
            float[] result = new float[length];

            for (int i = 0; i < length; i++)
            {
                test[i] = Convert.ToSingle(i);
            }

            Marshal.Copy(test, 0, ptr, length);
            Marshal.Copy(ptr, result, 0, length);

            for (int i = 0; i < length; i++)
            {
                Assert.AreEqual(result[i], test[i]);
            }

            status = CudaRuntimeApi.cudaFreeHost(ptr);
            Assert.AreEqual(status, cudaError.cudaSuccess);
        }

        [DllImport("Kernel32", SetLastError = true, CallingConvention = CallingConvention.Cdecl)]
        public static extern bool VirtualLock(
            IntPtr lpAddress,
            ulong dwSize
        );

        [TestMethod]
        public void cudaHostRegister_cudaHostUnRegister_test()
        {
            var length = 1024 * 1024;
            var size = 1024 * 1024 * sizeof(float);
            IntPtr ptr = Marshal.AllocHGlobal(size);

            Assert.IsFalse(VirtualLock(ptr, (ulong)size));

            var status = CudaRuntimeApi.cudaHostRegister(ptr, (ulong)size, DriverTypes.cudaHostRegisterDefault);
            Assert.AreEqual(status, cudaError.cudaSuccess);
            Console.WriteLine($"ptr : {ptr}");

            float[] test = new float[length];
            float[] result = new float[length];

            for (int i = 0; i < length; i++)
            {
                test[i] = Convert.ToSingle(i);
            }

            Marshal.Copy(test, 0, ptr, length);
            Marshal.Copy(ptr, result, 0, length);

            for (int i = 0; i < length; i++)
            {
                Assert.AreEqual(result[i], test[i]);
            }

            status = CudaRuntimeApi.cudaHostUnregister(ptr);
            Assert.AreEqual(status, cudaError.cudaSuccess);

            Marshal.FreeHGlobal(ptr);
        }

        [TestMethod]
        public void cudaMemGetInfo_test()
        {
            ulong free = 0;
            ulong total = 0;
            var status = CudaRuntimeApi.cudaMemGetInfo(ref free, ref total);
            Assert.AreEqual(status, cudaError.cudaSuccess);
            Console.WriteLine($"free : {free}, total : {total}");
        }

        [TestMethod]
        public void cudaMalloc_cudaFree_cudaMemcpy_test_cudaMemset_test()
        {
            int length = 1024 * 2;
            byte testValue = 5;
            byte[] test = new byte[length];
            byte[] result = new byte[length];

            IntPtr d_ptr = IntPtr.Zero;
            var size = length * sizeof(byte);
            var status = CudaRuntimeApi.cudaMalloc(ref d_ptr, (ulong)size);
            Assert.AreEqual(status, cudaError.cudaSuccess);
            Console.WriteLine($"ptr : {d_ptr}");

            GCHandle gchTest = GCHandle.Alloc(test, GCHandleType.Pinned);
            GCHandle gchResult = GCHandle.Alloc(result, GCHandleType.Pinned);
            IntPtr h_ptrTest = Marshal.UnsafeAddrOfPinnedArrayElement(test, 0);
            IntPtr h_ptrResult = Marshal.UnsafeAddrOfPinnedArrayElement(result, 0);

            status = CudaRuntimeApi.cudaMemcpy(d_ptr, h_ptrTest, (ulong)size, cudaMemcpyKind.HostToDevice);
            Assert.AreEqual(status, cudaError.cudaSuccess);

            status = CudaRuntimeApi.cudaMemset(d_ptr, testValue, (ulong)size);
            Assert.AreEqual(status, cudaError.cudaSuccess);

            status = CudaRuntimeApi.cudaMemcpy(h_ptrResult, d_ptr, (ulong)size, cudaMemcpyKind.DeviceToHost);
            Assert.AreEqual(status, cudaError.cudaSuccess);

            for (int i = 0; i < length; i++)
            {
                Assert.AreEqual(result[i], testValue);
            }

            status = CudaRuntimeApi.cudaFree(d_ptr);
            Assert.AreEqual(status, cudaError.cudaSuccess);

            gchTest.Free();
            gchResult.Free();
        }

        [TestMethod]
        public void cudaStreamCreate_cudaStreamDestroy_test()
        {
            IntPtr stream = IntPtr.Zero;
            var status = CudaRuntimeApi.cudaStreamCreate(ref stream);
            Assert.AreEqual(status, cudaError.cudaSuccess);

            status = CudaRuntimeApi.cudaStreamDestroy(stream);
            Assert.AreEqual(status, cudaError.cudaSuccess);
        }
    }
}