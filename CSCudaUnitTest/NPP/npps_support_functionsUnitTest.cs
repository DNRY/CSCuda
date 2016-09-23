using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CSCuda.NPP;


namespace CSCudaUnitTest.NPP
{
    [TestClass]
    public class npps_support_functionsUnitTest
    {
        [TestMethod]
        public void nppsMalloc_32s_test()
        {
            int size = 1024;
            IntPtr ptr = Npps.nppsMalloc_32s(size);
            Console.WriteLine("{0:X}", ptr.ToInt64());

            Npps.nppsFree(ptr);
        }
    }
}
