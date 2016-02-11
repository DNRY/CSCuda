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
    public partial class Nppi
    {
        /// <summary>
        /// 32-bit floating point complex to 32-bit floating point magnitude.
        /// Converts complex-number pixel image to single channel image computing the result pixels as the magnitude of the complex values.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMagnitude_32fc32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 32-bit floating point complex to 32-bit floating point squared magnitude.
        /// Converts complex-number pixel image to single channel image computing the result pixels as the squared magnitude of the complex values.
        /// The squared magnitude is an itermediate result of magnitude computation and can thus be computed faster than actual magnitude. If magnitudes are required for sorting/comparing only, using this function instead of nppiMagnitude_32fc32f_C1R can be a worthwhile performance optimization.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMagnitudeSqr_32fc32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

    }
}