using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CSCuda.NPP;

namespace CSCudaUnitTest
{
    [TestClass]
    public class nppCoreUnitTest
    {
        [TestMethod]
        public void nppGetLibVersion_test()
        {
            NppLibraryVersion version = NppCore.nppGetLibVersion();

            Console.WriteLine("NPP version : {0}.{1}", version.major, version.minor);
        }

        [TestMethod]
        public void nppGetGpuComputeCapability_test()
        {
            NppGpuComputeCapability capability = NppCore.nppGetGpuComputeCapability();

            Console.WriteLine("{0}", capability.ToString());
        }

        [TestMethod]
        public void nppGetGpuNumSMs_test()
        {
            Console.WriteLine("num of SM : {0}", NppCore.nppGetGpuNumSMs());
        }

        [TestMethod]
        public void nppGetMaxThreadsPerBlock_test()
        {
            Console.WriteLine("Max threads/Block : {0}", NppCore.nppGetMaxThreadsPerBlock());
        }

        [TestMethod]
        public void nppGetMaxThreadsPerSM_test()
        {
            Console.WriteLine("Max threads/SM : {0}", NppCore.nppGetMaxThreadsPerSM());
        }

        [TestMethod]
        public void nppGetGpuName_test()
        {
            Console.WriteLine("GPU name : {0}", NppCore.nppGetGpuName());
        }
    }
}
