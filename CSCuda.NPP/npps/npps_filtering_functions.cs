using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Npp8u = System.Byte;
using Npp8s = System.SByte;
using Npp16u = System.UInt16;
using Npp16s = System.Int16;
using Npp32u = System.UInt32;
using Npp32s = System.Int32;
using Npp64u = System.UInt64;
using Npp64s = System.Int64;
using Npp32f = System.Single;
using Npp64f = System.Double;

namespace CSCuda.NPP
{
    public partial class Npps
    {
        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsIntegralGetBufferSize_32s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsIntegral_32s(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            IntPtr pDeviceBuffer);

    }
}