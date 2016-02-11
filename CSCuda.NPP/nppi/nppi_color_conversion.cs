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
        /// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed YUV color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYUV_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed YUV color conversion with alpha, not affecting alpha. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYUV_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YUV color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYUV_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YUV color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYUV_8u_C3P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned planar YUV color conversion with alpha. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYUV_8u_AC4P4R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned packed YUV color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYUV_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed BGR with alpha to 4 channel 8-bit unsigned packed YUV color conversion with alpha, not affecting alpha. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYUV_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar YUV color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYUV_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YUV color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYUV_8u_C3P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed BGR with alpha to 4 channel 8-bit unsigned planar YUV color conversion with alpha. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYUV_8u_AC4P4R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed YUV to 3 channel 8-bit unsigned packed RGB color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYUVToRGB_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit packed YUV with alpha to 4 channel 8-bit unsigned packed RGB color conversion with alpha, not affecting alpha. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYUVToRGB_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned planar RGB color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYUVToRGB_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned packed RGB color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYUVToRGB_8u_P3C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed YUV to 3 channel 8-bit unsigned packed BGR color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYUVToBGR_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit packed YUV with alpha to 4 channel 8-bit unsigned packed BGR color conversion with alpha, not affecting alpha. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYUVToBGR_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned planar BGR color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYUVToBGR_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YUV to 3 channel 8-bit unsigned packed BGR color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYUVToBGR_8u_P3C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed RGB to 2 channel 8-bit unsigned packed YUV422 color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYUV422_8u_C3C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YUV422 color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYUV422_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YUV422 color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYUV422_8u_C3P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed YUV422 to 3 channel 8-bit unsigned packed RGB color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYUV422ToRGB_8u_C2C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned planar RGB color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_planar_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYUV422ToRGB_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YUV422 to 3 channel 8-bit unsigned packed RGB color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYUV422ToRGB_8u_P3C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YUV422 to 4 channel 8-bit unsigned packed RGB color conversion with alpha.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYUV422ToRGB_8u_P3AC4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar RGB to 3 channel 8-bit unsigned planar YUV420 color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYUV420_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YUV420 color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYUV420_8u_C3P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned planar RGB color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_planar_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYUV420ToRGB_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed RGB color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYUV420ToRGB_8u_P3C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YUV420 to 4 channel 8-bit unsigned packed RGB color conversion with constant alpha (0xFF).
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYUV420ToRGB_8u_P3C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YUV420 to 4 channel 8-bit unsigned packed RGB color conversion with alpha.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYUV420ToRGB_8u_P3AC4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned planar NV21 to 4 channel 8-bit unsigned packed ARGB color conversion with constant alpha (0xFF).
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array (one for Y plane, one for VU plane).</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiNV21ToRGB_8u_P2C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]IntPtr[] pSrc,
            int rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned pacmed BGR with alpha to 3 channel 8-bit unsigned planar YUV420 color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYUV420_8u_AC4P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YUV420 to 3 channel 8-bit unsigned packed BGR color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYUV420ToBGR_8u_P3C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YUV420 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha (0xFF).
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYUV420ToBGR_8u_P3C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned planar NV21 to 4 channel 8-bit unsigned packed BGRA color conversion with constant alpha (0xFF).
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array (one for Y plane, one for VU plane).</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiNV21ToBGR_8u_P2C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]IntPtr[] pSrc,
            int rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed RGB to 3 channel unsigned 8-bit packed YCbCr color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYCbCr_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed RGB with alpha to 4 channel unsigned 8-bit packed YCbCr with alpha color conversion, not affecting alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYCbCr_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel planar 8-bit unsigned RGB to 3 channel planar 8-bit YCbCr color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYCbCr_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed RGB to 3 channel unsigned 8-bit planar YCbCr color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYCbCr_8u_C3P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed RGB with alpha to 3 channel 8-bit unsigned planar YCbCr color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYCbCr_8u_AC4P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed YCbCr to 3 channel 8-bit unsigned packed RGB color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCrToRGB_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed YCbCr with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion, not affecting alpha. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCrToRGB_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned planar RGB color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCrToRGB_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed RGB color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCrToRGB_8u_P3C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr to 4 channel 8-bit unsigned packed RGB color conversion with constant alpha.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nAval">8-bit unsigned alpha constant.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCrToRGB_8u_P3C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp8u nAval);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed BGR color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCrToBGR_8u_P3C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nAval">8-bit unsigned alpha constant.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCrToBGR_8u_P3C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp8u nAval);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr to 3 channel 8-bit unsigned packed BGR_709CSC color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCrToBGR_709CSC_8u_P3C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr to 4 channel 8-bit unsigned packed BGR_709CSC color conversion with constant alpha.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nAval">8-bit unsigned alpha constant.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCrToBGR_709CSC_8u_P3C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp8u nAval);

        /// <summary>
        /// 3 channel 8-bit unsigned packed RGB to 2 channel 8-bit unsigned packed YCbCr422 color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYCbCr422_8u_C3C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr422 color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYCbCr422_8u_C3P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar RGB to 2 channel 8-bit unsigned packed YCbCr422 color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYCbCr422_8u_P3C2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned packed RGB color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr422ToRGB_8u_C2C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar RGB color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr422ToRGB_8u_C2P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed RGB color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr422ToRGB_8u_P3C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed RGB to 2 channel 8-bit unsigned packed YCrCb422 color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYCrCb422_8u_C3C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar RGB to 2 channel 8-bit unsigned packed YCrCb422 color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYCrCb422_8u_P3C2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned packed RGB color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCrCb422ToRGB_8u_C2C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar RGB color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCrCb422ToRGB_8u_C2P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed BGR to 2 channel 8-bit unsigned packed YCrCb422 color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYCbCr422_8u_C3C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed BGR with alpha to 2 channel 8-bit unsigned packed YCrCb422 color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYCbCr422_8u_AC4C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr422 color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYCbCr422_8u_C3P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr422 color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYCbCr422_8u_AC4P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned packed BGR color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr422ToBGR_8u_C2C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed YCrCb422 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nAval">8-bit unsigned alpha constant.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr422ToBGR_8u_C2C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp8u nAval);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned packed BGR color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr422ToBGR_8u_P3C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed RGB to 2 channel 8-bit unsigned packed CbYCr422 color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToCbYCr422_8u_C3C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed RGB first gets forward gamma corrected then converted to 2 channel 8-bit unsigned packed CbYCr422 color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToCbYCr422Gamma_8u_C3C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed CbYCrC22 to 3 channel 8-bit unsigned packed RGB color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCbYCr422ToRGB_8u_C2C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed BGR with alpha to 2 channel 8-bit unsigned packed CbYCr422 color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToCbYCr422_8u_AC4C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed BGR to 2 channel 8-bit unsigned packed CbYCr422_709HDTV color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToCbYCr422_709HDTV_8u_C3C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed BGR with alpha to 2 channel 8-bit unsigned packed CbYCr422_709HDTV color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToCbYCr422_709HDTV_8u_AC4C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed CbYCr422 to 4 channel 8-bit unsigned packed BGR color conversion with alpha. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nAval">8-bit unsigned alpha constant.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCbYCr422ToBGR_8u_C2C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp8u nAval);

        /// <summary>
        /// 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned packed BGR_709HDTV color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCbYCr422ToBGR_709HDTV_8u_C2C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed CbYCr422 to 4 channel 8-bit unsigned packed BGR_709HDTV color conversion with constant alpha. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nAval">8-bit unsigned alpha constant.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCbYCr422ToBGR_709HDTV_8u_C2C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp8u nAval);

        /// <summary>
        /// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned planar YCbCr420 color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYCbCr420_8u_C3P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed RGB color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr420ToRGB_8u_P3C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed RGB with alpha to 3 channel 8-bit unsigned planar YCrCb420 color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYCrCb420_8u_AC4P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCrCb420 to 4 channel 8-bit unsigned packed RGB color conversion with constant alpha.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nAval">8-bit unsigned alpha constant.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCrCb420ToRGB_8u_P3C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp8u nAval);

        /// <summary>
        /// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr420 color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYCbCr420_8u_C3P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr420 color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYCbCr420_8u_AC4P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr420_709CSC color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYCbCr420_709CSC_8u_C3P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr420_709CSC color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYCbCr420_709CSC_8u_AC4P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr420_709HDTV color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYCbCr420_709HDTV_8u_AC4P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCrCb420_709CSC color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYCrCb420_709CSC_8u_C3P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCrCb420_709CSC color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYCrCb420_709CSC_8u_AC4P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed BGR color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr420ToBGR_8u_P3C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr420 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nAval">8-bit unsigned alpha constant.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr420ToBGR_8u_P3C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp8u nAval);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned packed BGR_709CSC color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr420ToBGR_709CSC_8u_P3C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr420 to 4 channel 8-bit unsigned packed BGR_709HDTV color conversion with constant alpha.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nAval">8-bit unsigned alpha constant.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr420ToBGR_709HDTV_8u_P3C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp8u nAval);

        /// <summary>
        /// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCrCb420 color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYCrCb420_8u_C3P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCrCb420 color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYCrCb420_8u_AC4P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr411 color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYCbCr411_8u_C3P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr411 color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYCbCr411_8u_AC4P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar YCbCr color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYCbCr_8u_C3P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed BGR with alpha to 3 channel 8-bit unsigned planar YCbCr color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYCbCr_8u_AC4P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed BGR with alpha to 4 channel 8-bit unsigned planar YCbCr color conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToYCbCr_8u_AC4P4R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned packed BGR color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr411ToBGR_8u_P3C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr411 to 4 channel 8-bit unsigned packed BGR color conversion with constant alpha.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nAval">8-bit unsigned alpha constant.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr411ToBGR_8u_P3C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp8u nAval);

        /// <summary>
        /// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed XYZ color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToXYZ_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed XYZ with alpha color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToXYZ_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed XYZ to 3 channel 8-bit unsigned packed RGB color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXYZToRGB_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed XYZ with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXYZToRGB_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed LUV color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToLUV_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed LUV with alpha color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToLUV_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed LUV to 3 channel 8-bit unsigned packed RGB color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUVToRGB_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed LUV with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUVToRGB_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned packed Lab color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToLab_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed Lab to 3 channel 8-bit unsigned packed BGR color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLabToBGR_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed YCC color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYCC_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed YCC with alpha color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToYCC_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed YCC to 3 channel 8-bit unsigned packed RGB color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCCToRGB_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed YCC with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCCToRGB_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed HLS color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToHLS_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed HLS with alpha color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToHLS_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed HLS to 3 channel 8-bit unsigned packed RGB color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiHLSToRGB_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed HLS with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiHLSToRGB_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed BGR with alpha to 4 channel 8-bit unsigned packed HLS with alpha color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToHLS_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed BGR to 3 channel 8-bit unsigned planar HLS color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToHLS_8u_C3P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed BGR with alpha to 4 channel 8-bit unsigned planar HLS with alpha color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToHLS_8u_AC4P4R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned packed HLS color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToHLS_8u_P3C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned planar BGR with alpha to 4 channel 8-bit unsigned packed HLS with alpha color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToHLS_8u_AP4C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar BGR to 3 channel 8-bit unsigned planar HLS color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToHLS_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned planar BGR with alpha to 4 channel 8-bit unsigned planar HLS with alpha color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiBGRToHLS_8u_AP4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed HLS to 3 channel 8-bit unsigned planar BGR color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiHLSToBGR_8u_C3P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed HLS with alpha to 4 channel 8-bit unsigned planar BGR with alpha color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiHLSToBGR_8u_AC4P4R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar HLS to 3 channel 8-bit unsigned planar BGR color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiHLSToBGR_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned planar HLS with alpha to 4 channel 8-bit unsigned planar BGR with alpha color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiHLSToBGR_8u_AP4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed HLS with alpha to 4 channel 8-bit unsigned packed BGR with alpha color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiHLSToBGR_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar HLS to 3 channel 8-bit unsigned packed BGR color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiHLSToBGR_8u_P3C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned planar HLS with alpha to 4 channel 8-bit unsigned packed BGR with alpha color conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiHLSToBGR_8u_AP4C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed RGB to 3 channel 8-bit unsigned packed HSV color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToHSV_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed RGB with alpha to 4 channel 8-bit unsigned packed HSV with alpha color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToHSV_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed HSV to 3 channel 8-bit unsigned packed RGB color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiHSVToRGB_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed HSV with alpha to 4 channel 8-bit unsigned packed RGB with alpha color conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiHSVToRGB_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed RGB to 1 channel 8-bit unsigned packed Gray conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToGray_8u_C3C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed RGB with alpha to 1 channel 8-bit unsigned packed Gray conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToGray_8u_AC4C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 16-bit unsigned packed RGB to 1 channel 16-bit unsigned packed Gray conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToGray_16u_C3C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 16-bit unsigned packed RGB with alpha to 1 channel 16-bit unsigned packed Gray conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToGray_16u_AC4C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 16-bit signed packed RGB to 1 channel 16-bit signed packed Gray conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToGray_16s_C3C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 16-bit signed packed RGB with alpha to 1 channel 16-bit signed packed Gray conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToGray_16s_AC4C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 32-bit floating point packed RGB to 1 channel 32-bit floating point packed Gray conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToGray_32f_C3C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 32-bit floating point packed RGB with alpha to 1 channel 32-bit floating point packed Gray conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRGBToGray_32f_AC4C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed RGB to 1 channel 8-bit unsigned packed Gray conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aCoeffs">fixed size array of constant floating point conversion coefficient values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorToGray_8u_C3C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aCoeffs);

        /// <summary>
        /// 4 channel 8-bit unsigned packed RGB with alpha to 1 channel 8-bit unsigned packed Gray conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aCoeffs">fixed size array of constant floating point conversion coefficient values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorToGray_8u_AC4C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aCoeffs);

        /// <summary>
        /// 4 channel 8-bit unsigned packed RGBA to 1 channel 8-bit unsigned packed Gray conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aCoeffs">fixed size array of constant floating point conversion coefficient values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorToGray_8u_C4C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32f[] aCoeffs);

        /// <summary>
        /// 3 channel 16-bit unsigned packed RGB to 1 channel 16-bit unsigned packed Gray conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aCoeffs">fixed size array of constant floating point conversion coefficient values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorToGray_16u_C3C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aCoeffs);

        /// <summary>
        /// 4 channel 16-bit unsigned packed RGB with alpha to 1 channel 16-bit unsigned packed Gray conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aCoeffs">fixed size array of constant floating point conversion coefficient values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorToGray_16u_AC4C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aCoeffs);

        /// <summary>
        /// 4 channel 16-bit unsigned packed RGBA to 1 channel 16-bit unsigned packed Gray conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aCoeffs">fixed size array of constant floating point conversion coefficient values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorToGray_16u_C4C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32f[] aCoeffs);

        /// <summary>
        /// 3 channel 16-bit signed packed RGB to 1 channel 16-bit signed packed Gray conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aCoeffs">fixed size array of constant floating point conversion coefficient values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorToGray_16s_C3C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aCoeffs);

        /// <summary>
        /// 4 channel 16-bit signed packed RGB with alpha to 1 channel 16-bit signed packed Gray conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aCoeffs">fixed size array of constant floating point conversion coefficient values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorToGray_16s_AC4C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aCoeffs);

        /// <summary>
        /// 4 channel 16-bit signed packed RGBA to 1 channel 16-bit signed packed Gray conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aCoeffs">fixed size array of constant floating point conversion coefficient values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorToGray_16s_C4C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32f[] aCoeffs);

        /// <summary>
        /// 3 channel 32-bit floating point packed RGB to 1 channel 32-bit floating point packed Gray conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aCoeffs">fixed size array of constant floating point conversion coefficient values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorToGray_32f_C3C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aCoeffs);

        /// <summary>
        /// 4 channel 32-bit floating point packed RGB with alpha to 1 channel 32-bit floating point packed Gray conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aCoeffs">fixed size array of constant floating point conversion coefficient values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorToGray_32f_AC4C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aCoeffs);

        /// <summary>
        /// 4 channel 32-bit floating point packed RGBA to 1 channel 32-bit floating point packed Gray conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aCoeffs">fixed size array of constant floating point conversion coefficient values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorToGray_32f_C4C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32f[] aCoeffs);

        /// <summary>
        /// 1 channel 8-bit unsigned packed CFA grayscale Bayer pattern to 3 channel 8-bit unsigned packed RGB conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">full source image width and height relative to pSrc.</param>
        /// <param name="oSrcROI">rectangle specifying starting source image pixel x and y location relative to pSrc and ROI width and height.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="eGrid">enumeration value specifying bayer grid registration position at location oSrcROI.x, oSrcROI.y relative to pSrc.</param>
        /// <param name="eInterpolation">MUST be NPPI_INTER_UNDEFINED</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCFAToRGB_8u_C1C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiBayerGridPosition eGrid,
            NppiInterpolationMode eInterpolation);

        /// <summary>
        /// 1 channel 8-bit unsigned packed CFA grayscale Bayer pattern to 4 channel 8-bit unsigned packed RGB conversion with alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">full source image width and height relative to pSrc.</param>
        /// <param name="oSrcROI">rectangle specifying starting source image pixel x and y location relative to pSrc and ROI width and height.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="eGrid">enumeration value specifying bayer grid registration position at location oSrcROI.x, oSrcROI.y relative to pSrc.</param>
        /// <param name="eInterpolation">MUST be NPPI_INTER_UNDEFINED</param>
        /// <param name="nAlpha">constant alpha value to be written to each destination pixel</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCFAToRGBA_8u_C1AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiBayerGridPosition eGrid,
            NppiInterpolationMode eInterpolation,
            Npp8u nAlpha);

        /// <summary>
        /// 1 channel 16-bit unsigned packed CFA grayscale Bayer pattern to 3 channel 16-bit unsigned packed RGB conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">full source image width and height relative to pSrc.</param>
        /// <param name="oSrcROI">rectangle specifying starting source image pixel x and y location relative to pSrc and ROI width and height.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="eGrid">enumeration value specifying bayer grid registration position at location oSrcROI.x, oSrcROI.y relative to pSrc.</param>
        /// <param name="eInterpolation">MUST be NPPI_INTER_UNDEFINED</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCFAToRGB_16u_C1C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiBayerGridPosition eGrid,
            NppiInterpolationMode eInterpolation);

        /// <summary>
        /// 1 channel 16-bit unsigned packed CFA grayscale Bayer pattern to 4 channel 16-bit unsigned packed RGB conversion with alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">full source image width and height relative to pSrc.</param>
        /// <param name="oSrcROI">rectangle specifying starting source image pixel x and y location relative to pSrc and ROI width and height.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="eGrid">enumeration value specifying bayer grid registration position at location oSrcROI.x, oSrcROI.y relative to pSrc.</param>
        /// <param name="eInterpolation">MUST be NPPI_INTER_UNDEFINED</param>
        /// <param name="nAlpha">constant alpha value to be written to each destination pixel</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCFAToRGBA_16u_C1AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiRect oSrcROI,
            IntPtr pDst,
            int nDstStep,
            NppiBayerGridPosition eGrid,
            NppiInterpolationMode eInterpolation,
            Npp16u nAlpha);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr420 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDstY">destination_planar_image_pointer.</param>
        /// <param name="nDstYStep">destination_planar_image_line_step.</param>
        /// <param name="pDstCbCr">destination_planar_image_pointer.</param>
        /// <param name="nDstCbCrStep">destination_planar_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr420ToYCbCr411_8u_P3P2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDstY,
            int nDstYStep,
            IntPtr pDstCbCr,
            int nDstCbCrStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
        /// </summary>
        /// <param name="pSrcY">source_planar_image_pointer.</param>
        /// <param name="nSrcYStep">source_planar_image_line_step.</param>
        /// <param name="pSrcCbCr">source_planar_image_pointer.</param>
        /// <param name="nSrcCbCrStep">source_planar_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr420ToYCbCr411_8u_P2P3R(
            IntPtr pSrcY,
            int nSrcYStep,
            IntPtr pSrcCbCr,
            int nSrcCbCrStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr422_8u_C2P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr422 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr422_8u_P3C2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned packed YCrCb422 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr422ToYCrCb422_8u_C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr422 to 2 channel 8-bit unsigned packed YCrCb422 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr422ToYCrCb422_8u_P3C2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned packed CbYCr422 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr422ToCbYCr422_8u_C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCbYCr422ToYCbCr411_8u_C2P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr422ToYCbCr420_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr422 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDstY">destination_planar_image_pointer.</param>
        /// <param name="nDstYStep">destination_planar_image_line_step.</param>
        /// <param name="pDstCbCr">destination_planar_image_pointer.</param>
        /// <param name="nDstCbCrStep">destination_planar_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr422ToYCbCr420_8u_P3P2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDstY,
            int nDstYStep,
            IntPtr pDstCbCr,
            int nDstCbCrStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr422ToYCbCr420_8u_C2P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDstY">destination_planar_image_pointer.</param>
        /// <param name="nDstYStep">destination_planar_image_line_step.</param>
        /// <param name="pDstCbCr">destination_planar_image_pointer.</param>
        /// <param name="nDstCbCrStep">destination_planar_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr422ToYCbCr420_8u_C2P2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDstY,
            int nDstYStep,
            IntPtr pDstCbCr,
            int nDstCbCrStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCrCb420 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCrCb420ToYCbCr422_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCrCb420 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCrCb420ToYCbCr422_8u_P3C2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCrCb420 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr422ToYCrCb420_8u_C2P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr422 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr422ToYCbCr411_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr422 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDstY">destination_planar_image_pointer.</param>
        /// <param name="nDstYStep">destination_planar_image_line_step.</param>
        /// <param name="pDstCbCr">destination_planar_image_pointer.</param>
        /// <param name="nDstCbCrStep">destination_planar_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr422ToYCbCr411_8u_P3P2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDstY,
            int nDstYStep,
            IntPtr pDstCbCr,
            int nDstCbCrStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed YCbCr422 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr422ToYCbCr411_8u_C2P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed YCbCr422 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDstY">destination_planar_image_pointer.</param>
        /// <param name="nDstYStep">destination_planar_image_line_step.</param>
        /// <param name="pDstCbCr">destination_planar_image_pointer.</param>
        /// <param name="nDstCbCrStep">destination_planar_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr422ToYCbCr411_8u_C2P2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDstY,
            int nDstYStep,
            IntPtr pDstCbCr,
            int nDstCbCrStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCrCb422ToYCbCr422_8u_C2P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCrCb422ToYCbCr420_8u_C2P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed YCrCb422 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCrCb422ToYCbCr411_8u_C2P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed CbYCr422 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCbYCr422ToYCbCr422_8u_C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCbYCr422ToYCbCr422_8u_C2P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCbYCr422ToYCbCr420_8u_C2P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed CbYCr422 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDstY">destination_planar_image_pointer.</param>
        /// <param name="nDstYStep">destination_planar_image_line_step.</param>
        /// <param name="pDstCbCr">destination_planar_image_pointer.</param>
        /// <param name="nDstCbCrStep">destination_planar_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCbYCr422ToYCbCr420_8u_C2P2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDstY,
            int nDstYStep,
            IntPtr pDstCbCr,
            int nDstCbCrStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned packed CbYCr422 to 3 channel 8-bit unsigned planar YCrCb420 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCbYCr422ToYCrCb420_8u_C2P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr420 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDstY">destination_planar_image_pointer.</param>
        /// <param name="nDstYStep">destination_planar_image_line_step.</param>
        /// <param name="pDstCbCr">destination_planar_image_pointer.</param>
        /// <param name="nDstCbCrStep">destination_planar_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr420_8u_P3P2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDstY,
            int nDstYStep,
            IntPtr pDstCbCr,
            int nDstCbCrStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
        /// </summary>
        /// <param name="pSrcY">source_planar_image_pointer.</param>
        /// <param name="nSrcYStep">source_planar_image_line_step.</param>
        /// <param name="pSrcCbCr">source_planar_image_pointer.</param>
        /// <param name="nSrcCbCrStep">source_planar_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr420_8u_P2P3R(
            IntPtr pSrcY,
            int nSrcYStep,
            IntPtr pSrcCbCr,
            int nSrcCbCrStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr420ToYCbCr422_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
        /// </summary>
        /// <param name="pSrcY">source_planar_image_pointer.</param>
        /// <param name="nSrcYStep">source_planar_image_line_step.</param>
        /// <param name="pSrcCbCr">source_planar_image_pointer.</param>
        /// <param name="nSrcCbCrStep">source_planar_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr420ToYCbCr422_8u_P2P3R(
            IntPtr pSrcY,
            int nSrcYStep,
            IntPtr pSrcCbCr,
            int nSrcCbCrStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned planar YCbCr420 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
        /// </summary>
        /// <param name="pSrcY">source_planar_image_pointer.</param>
        /// <param name="nSrcYStep">source_planar_image_line_step.</param>
        /// <param name="pSrcCbCr">source_planar_image_pointer.</param>
        /// <param name="nSrcCbCrStep">source_planar_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr420ToYCbCr422_8u_P2C2R(
            IntPtr pSrcY,
            int nSrcYStep,
            IntPtr pSrcCbCr,
            int nSrcCbCrStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned planar YCbCr420 to 2 channel 8-bit unsigned packed CbYCr422 sampling format conversion.
        /// </summary>
        /// <param name="pSrcY">source_planar_image_pointer.</param>
        /// <param name="nSrcYStep">source_planar_image_line_step.</param>
        /// <param name="pSrcCbCr">source_planar_image_pointer.</param>
        /// <param name="nSrcCbCrStep">source_planar_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr420ToCbYCr422_8u_P2C2R(
            IntPtr pSrcY,
            int nSrcYStep,
            IntPtr pSrcCbCr,
            int nSrcCbCrStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned planar YCbCr420 to 3 channel 8-bit unsigned planar YCrCb420 sampling format conversion.
        /// </summary>
        /// <param name="pSrcY">source_planar_image_pointer.</param>
        /// <param name="nSrcYStep">source_planar_image_line_step.</param>
        /// <param name="pSrcCbCr">source_planar_image_pointer.</param>
        /// <param name="nSrcCbCrStep">source_planar_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr420ToYCrCb420_8u_P2P3R(
            IntPtr pSrcY,
            int nSrcYStep,
            IntPtr pSrcCbCr,
            int nSrcCbCrStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCrCb420 to 2 channel 8-bit unsigned packed CbYCr422 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCrCb420ToCbYCr422_8u_P3C2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCrCb420 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDstY">destination_planar_image_pointer.</param>
        /// <param name="nDstYStep">destination_planar_image_line_step.</param>
        /// <param name="pDstCbCr">destination_planar_image_pointer.</param>
        /// <param name="nDstCbCrStep">destination_planar_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCrCb420ToYCbCr420_8u_P3P2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDstY,
            int nDstYStep,
            IntPtr pDstCbCr,
            int nDstCbCrStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCrCb420 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDstY">destination_planar_image_pointer.</param>
        /// <param name="nDstYStep">destination_planar_image_line_step.</param>
        /// <param name="pDstCbCr">destination_planar_image_pointer.</param>
        /// <param name="nDstCbCrStep">destination_planar_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCrCb420ToYCbCr411_8u_P3P2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDstY,
            int nDstYStep,
            IntPtr pDstCbCr,
            int nDstCbCrStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned planar YCbCr411 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDstY">destination_planar_image_pointer.</param>
        /// <param name="nDstYStep">destination_planar_image_line_step.</param>
        /// <param name="pDstCbCr">destination_planar_image_pointer.</param>
        /// <param name="nDstCbCrStep">destination_planar_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr411_8u_P3P2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDstY,
            int nDstYStep,
            IntPtr pDstCbCr,
            int nDstCbCrStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr411 sampling format conversion.
        /// </summary>
        /// <param name="pSrcY">source_planar_image_pointer.</param>
        /// <param name="nSrcYStep">source_planar_image_line_step.</param>
        /// <param name="pSrcCbCr">source_planar_image_pointer.</param>
        /// <param name="nSrcCbCrStep">source_planar_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr411_8u_P2P3R(
            IntPtr pSrcY,
            int nSrcYStep,
            IntPtr pSrcCbCr,
            int nSrcCbCrStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr411ToYCbCr422_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr411ToYCbCr422_8u_P3C2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr422 sampling format conversion.
        /// </summary>
        /// <param name="pSrcY">source_planar_image_pointer.</param>
        /// <param name="nSrcYStep">source_planar_image_line_step.</param>
        /// <param name="pSrcCbCr">source_planar_image_pointer.</param>
        /// <param name="nSrcCbCrStep">source_planar_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr411ToYCbCr422_8u_P2P3R(
            IntPtr pSrcY,
            int nSrcYStep,
            IntPtr pSrcCbCr,
            int nSrcCbCrStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned packed YCbCr422 sampling format conversion.
        /// </summary>
        /// <param name="pSrcY">source_planar_image_pointer.</param>
        /// <param name="nSrcYStep">source_planar_image_line_step.</param>
        /// <param name="pSrcCbCr">source_planar_image_pointer.</param>
        /// <param name="nSrcCbCrStep">source_planar_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr411ToYCbCr422_8u_P2C2R(
            IntPtr pSrcY,
            int nSrcYStep,
            IntPtr pSrcCbCr,
            int nSrcCbCrStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCrCb422 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr411ToYCrCb422_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned packed YCrCb422 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr411ToYCrCb422_8u_P3C2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr411ToYCbCr420_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar YCbCr411 to 2 channel 8-bit unsigned planar YCbCr420 sampling format conversion. images.
        /// </summary>
        /// <param name="pSrc">source_planar_image_pointer_array.</param>
        /// <param name="rSrcStep">source_planar_image_line_step_array.</param>
        /// <param name="pDstY">destination_planar_image_pointer.</param>
        /// <param name="nDstYStep">destination_planar_image_line_step.</param>
        /// <param name="pDstCbCr">destination_planar_image_pointer.</param>
        /// <param name="nDstCbCrStep">destination_planar_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr411ToYCbCr420_8u_P3P2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rSrcStep,
            IntPtr pDstY,
            int nDstYStep,
            IntPtr pDstCbCr,
            int nDstCbCrStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCbCr420 sampling format conversion.
        /// </summary>
        /// <param name="pSrcY">source_planar_image_pointer.</param>
        /// <param name="nSrcYStep">source_planar_image_line_step.</param>
        /// <param name="pSrcCbCr">source_planar_image_pointer.</param>
        /// <param name="nSrcCbCrStep">source_planar_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr411ToYCbCr420_8u_P2P3R(
            IntPtr pSrcY,
            int nSrcYStep,
            IntPtr pSrcCbCr,
            int nSrcCbCrStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned planar YCbCr411 to 3 channel 8-bit unsigned planar YCrCb420 sampling format conversion.
        /// </summary>
        /// <param name="pSrcY">source_planar_image_pointer.</param>
        /// <param name="nSrcYStep">source_planar_image_line_step.</param>
        /// <param name="pSrcCbCr">source_planar_image_pointer.</param>
        /// <param name="nSrcCbCrStep">source_planar_image_line_step.</param>
        /// <param name="pDst">destination_planar_image_pointer_array.</param>
        /// <param name="rDstStep">destination_planar_image_line_step_array.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiYCbCr411ToYCrCb420_8u_P2P3R(
            IntPtr pSrcY,
            int nSrcYStep,
            IntPtr pSrcCbCr,
            int nSrcCbCrStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] rDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed color not in place forward gamma correction.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGammaFwd_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed color in place forward gamma correction.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGammaFwd_8u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed color with alpha not in place forward gamma correction.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGammaFwd_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed color with alpha in place forward gamma correction.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGammaFwd_8u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar color not in place forward gamma correction.
        /// </summary>
        /// <param name="pSrc">source planar pixel format image pointer array.</param>
        /// <param name="nSrcStep">source planar pixel format image line step.</param>
        /// <param name="pDst">destination planar pixel format image pointer array.</param>
        /// <param name="nDstStep">destination planar pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGammaFwd_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar color in place forward gamma correction.
        /// </summary>
        /// <param name="pSrcDst">in place planar pixel format image pointer array.</param>
        /// <param name="nSrcDstStep">in place planar pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGammaFwd_8u_IP3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed color not in place inverse gamma correction.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGammaInv_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned packed color in place inverse gamma correction.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGammaInv_8u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed color with alpha not in place inverse gamma correction.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGammaInv_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned packed color with alpha in place inverse gamma correction.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGammaInv_8u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar color not in place inverse gamma correction.
        /// </summary>
        /// <param name="pSrc">source planar pixel format image pointer array.</param>
        /// <param name="nSrcStep">source planar pixel format image line step.</param>
        /// <param name="pDst">destination planar pixel format image pointer array.</param>
        /// <param name="nDstStep">destination planar pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGammaInv_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned planar color in place inverse gamma correction.
        /// </summary>
        /// <param name="pSrcDst">in place planar pixel format image pointer array.</param>
        /// <param name="nSrcDstStep">in place planar pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiGammaInv_8u_IP3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 1 channel 8-bit unsigned packed color complement color key replacement of source image 1 by source image 2.
        /// </summary>
        /// <param name="pSrc1">source1 packed pixel format image pointer.</param>
        /// <param name="nSrc1Step">source1 packed pixel format image line step.</param>
        /// <param name="pSrc2">source2 packed pixel format image pointer.</param>
        /// <param name="nSrc2Step">source2 packed pixel format image line step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nColorKeyConst">color key constant</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCompColorKey_8u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp8u nColorKeyConst);

        /// <summary>
        /// 3 channel 8-bit unsigned packed color complement color key replacement of source image 1 by source image 2.
        /// </summary>
        /// <param name="pSrc1">source1 packed pixel format image pointer.</param>
        /// <param name="nSrc1Step">source1 packed pixel format image line step.</param>
        /// <param name="pSrc2">source2 packed pixel format image pointer.</param>
        /// <param name="nSrc2Step">source2 packed pixel format image line step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nColorKeyConst">color key constant array</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCompColorKey_8u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] nColorKeyConst);

        /// <summary>
        /// 4 channel 8-bit unsigned packed color complement color key replacement of source image 1 by source image 2.
        /// </summary>
        /// <param name="pSrc1">source1 packed pixel format image pointer.</param>
        /// <param name="nSrc1Step">source1 packed pixel format image line step.</param>
        /// <param name="pSrc2">source2 packed pixel format image pointer.</param>
        /// <param name="nSrc2Step">source2 packed pixel format image line step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nColorKeyConst">color key constant array</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCompColorKey_8u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp8u[] nColorKeyConst);

        /// <summary>
        /// 4 channel 8-bit unsigned packed color complement color key replacement of source image 1 by source image 2 with alpha blending.
        /// </summary>
        /// <param name="pSrc1">source1 packed pixel format image pointer.</param>
        /// <param name="nSrc1Step">source1 packed pixel format image line step.</param>
        /// <param name="nAlpha1">source1 image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pSrc2">source2 packed pixel format image pointer.</param>
        /// <param name="nSrc2Step">source2 packed pixel format image line step.</param>
        /// <param name="nAlpha2">source2 image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nColorKeyConst">color key constant array</param>
        /// <param name="nppAlphaOp">NppiAlphaOp alpha compositing operation selector (excluding premul ops).</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaCompColorKey_8u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp8u nAlpha1,
            IntPtr pSrc2,
            int nSrc2Step,
            Npp8u nAlpha2,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp8u[] nColorKeyConst,
            NppiAlphaOp nppAlphaOp);

        /// <summary>
        /// 1 channel 8-bit unsigned color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 1 channel 8-bit unsigned in place color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 2 channel 8-bit unsigned color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8u_C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 2 channel 8-bit unsigned in place color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8u_C2IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 3 channel 8-bit unsigned color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 3 channel 8-bit unsigned in place color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 4 channel 8-bit unsigned color twist, with alpha copy.
        /// An input color twist matrix with floating-point coefficient values is applied with in ROI. Alpha channel is the last channel and is copied unmodified from the source pixel to the destination pixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 4 channel 8-bit unsigned in place color twist, not affecting Alpha.
        /// An input color twist matrix with floating-point coefficient values is applied with in ROI. Alpha channel is the last channel and is unmodified.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8u_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 4 channel 8-bit unsigned color twist, not affecting Alpha.
        /// An input color twist matrix with floating-point coefficient values is applied with in ROI. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 4 channel 8-bit unsigned in place color twist, not affecting Alpha.
        /// An input color twist matrix with floating-point coefficient values is applied with in ROI. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 4 channel 8-bit unsigned color twist with 4x4 matrix and constant vector addition.
        /// An input 4x4 color twist matrix with floating-point coefficient values with an additional constant vector addition is applied within ROI. For this particular version of the function the result is generated as shown below.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32fC_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32f[] aTwist,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32f[] aConstants);

        /// <summary>
        /// 4 channel 8-bit unsigned in place color twist with 4x4 matrix and an additional constant vector addition.
        /// An input 4x4 color twist matrix with floating-point coefficient values with an additional constant vector addition is applied within ROI. For this particular version of the function the result is generated as shown below.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32fC_8u_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32f[] aTwist,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32f[] aConstants);

        /// <summary>
        /// 3 channel 8-bit unsigned planar color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 3 channel 8-bit unsigned planar in place color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrcDst">in place planar pixel format image pointer array, one pointer per plane.</param>
        /// <param name="nSrcDstStep">in place planar pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8u_IP3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 1 channel 8-bit signed color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 1 channel 8-bit signed in place color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8s_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 2 channel 8-bit signed color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8s_C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 2 channel 8-bit signed in place color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8s_C2IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 3 channel 8-bit signed color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 3 channel 8-bit signed in place color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8s_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 4 channel 8-bit signed color twist, with alpha copy.
        /// An input color twist matrix with floating-point coefficient values is applied with in ROI. Alpha channel is the last channel and is copied unmodified from the source pixel to the destination pixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 4 channel 8-bit signed in place color twist, not affecting Alpha.
        /// An input color twist matrix with floating-point coefficient values is applied with in ROI. Alpha channel is the last channel and is unmodified.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8s_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 4 channel 8-bit signed color twist, not affecting Alpha.
        /// An input color twist matrix with floating-point coefficient values is applied with in ROI. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 4 channel 8-bit signed in place color twist, not affecting Alpha.
        /// An input color twist matrix with floating-point coefficient values is applied with in ROI. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8s_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 3 channel 8-bit signed planar color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8s_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 3 channel 8-bit signed planar in place color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrcDst">in place planar pixel format image pointer array, one pointer per plane.</param>
        /// <param name="nSrcDstStep">in place planar pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_8s_IP3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 1 channel 16-bit unsigned color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 1 channel 16-bit unsigned in place color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_16u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 2 channel 16-bit unsigned color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_16u_C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 2 channel 16-bit unsigned in place color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_16u_C2IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 3 channel 16-bit unsigned color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 3 channel 16-bit unsigned in place color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_16u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 4 channel 16-bit unsigned color twist, not affecting Alpha.
        /// An input color twist matrix with floating-point coefficient values is applied with in ROI. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 4 channel 16-bit unsigned in place color twist, not affecting Alpha.
        /// An input color twist matrix with floating-point coefficient values is applied with in ROI. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_16u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 3 channel 16-bit unsigned planar color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_16u_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 3 channel 16-bit unsigned planar in place color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrcDst">in place planar pixel format image pointer array, one pointer per plane.</param>
        /// <param name="nSrcDstStep">in place planar pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_16u_IP3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 1 channel 16-bit signed color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 1 channel 16-bit signed in place color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_16s_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 2 channel 16-bit signed color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_16s_C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 2 channel 16-bit signed in place color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_16s_C2IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 3 channel 16-bit signed color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 3 channel 16-bit signed in place color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_16s_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 4 channel 16-bit signed color twist, not affecting Alpha.
        /// An input color twist matrix with floating-point coefficient values is applied with in ROI. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 4 channel 16-bit signed in place color twist, not affecting Alpha.
        /// An input color twist matrix with floating-point coefficient values is applied with in ROI. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_16s_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 3 channel 16-bit signed planar color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_16s_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 3 channel 16-bit signed planar in place color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrcDst">in place planar pixel format image pointer array, one pointer per plane.</param>
        /// <param name="nSrcDstStep">in place planar pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist32f_16s_IP3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 1 channel 32-bit floating point color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 1 channel 32-bit floating point in place color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist_32f_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 2 channel 32-bit floating point color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist_32f_C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 2 channel 32-bit floating point in place color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist_32f_C2IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 3 channel 32-bit floating point color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 3 channel 32-bit floating point in place color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist_32f_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 4 channel 32-bit floating point color twist, with alpha copy.
        /// An input color twist matrix with floating-point coefficient values is applied with in ROI. Alpha channel is the last channel and is copied unmodified from the source pixel to the destination pixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 4 channel 32-bit floating point in place color twist, not affecting Alpha.
        /// An input color twist matrix with floating-point coefficient values is applied with in ROI. Alpha channel is the last channel and is not modified.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist_32f_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 4 channel 32-bit floating point color twist, not affecting Alpha.
        /// An input color twist matrix with floating-point coefficient values is applied with in ROI. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 4 channel 32-bit floating point in place color twist, not affecting Alpha.
        /// An input color twist matrix with floating-point coefficient values is applied with in ROI. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist_32f_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 4 channel 32-bit floating point color twist with 4x4 matrix and constant vector addition.
        /// An input 4x4 color twist matrix with floating-point coefficient values with an additional constant vector addition is applied within ROI. For this particular version of the function the result is generated as shown below.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist_32fC_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32f[] aTwist,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32f[] aConstants);

        /// <summary>
        /// 4 channel 32-bit floating point in place color twist with 4x4 matrix and an additional constant vector addition.
        /// An input 4x4 color twist matrix with floating-point coefficient values with an additional constant vector addition is applied within ROI. For this particular version of the function the result is generated as shown below.
        /// </summary>
        /// <param name="pSrcDst">in place packed pixel format image pointer.</param>
        /// <param name="nSrcDstStep">in place packed pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist_32fC_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32f[] aTwist,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32f[] aConstants);

        /// <summary>
        /// 3 channel 32-bit floating point planar color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist_32f_P3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 3 channel 32-bit floating point planar in place color twist.
        /// An input color twist matrix with floating-point coefficient values is applied within ROI.
        /// </summary>
        /// <param name="pSrcDst">in place planar pixel format image pointer array, one pointer per plane.</param>
        /// <param name="nSrcDstStep">in place planar pixel format image line step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aTwist">The color twist matrix with floating-point coefficient values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiColorTwist_32f_IP3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aTwist);

        /// <summary>
        /// 8-bit unsigned look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points with no interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 8-bit unsigned look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points with no interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_8u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 3 channel 8-bit unsigned look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points using no interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 3 channel 8-bit unsigned look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points with no interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_8u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 8-bit unsigned look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points with no interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 8-bit unsigned look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points using no interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_8u_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 8-bit unsigned look-up-table color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points using no interpolation. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 8-bit unsigned look-up-table in place color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points using no interpolation. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_8u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 16-bit unsigned look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points with no interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 16-bit unsigned look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points with no interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_16u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 3 channel 16-bit unsigned look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points using no interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 3 channel 16-bit unsigned look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points with no interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_16u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit unsigned look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points with no interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit unsigned look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points using no interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_16u_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit unsigned look-up-table color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points using no interpolation. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit unsigned look-up-table in place color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points using no interpolation. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_16u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 16-bit signed look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points with no interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 16-bit signed look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points with no interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_16s_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 3 channel 16-bit signed look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points using no interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 3 channel 16-bit signed look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points with no interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_16s_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit signed look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points with no interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit signed look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points using no interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_16s_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit signed look-up-table color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points using no interpolation. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit signed look-up-table in place color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points using no interpolation. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_16s_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 32-bit floating point look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points with no interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 32-bit floating point look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points with no interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_32f_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 3 channel 32-bit floating point look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points using no interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 3 channel 32-bit floating point look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points with no interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_32f_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 32-bit floating point look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points with no interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 32-bit floating point look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points using no interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_32f_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 32-bit floating point look-up-table color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points using no interpolation. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 32-bit floating point look-up-table in place color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points using no interpolation. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_32f_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 8-bit unsigned linear interpolated look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points through linear interpolation.
        /// NOTE: As of the 5.0 release of NPP, the pValues and pLevels pointers need to be device memory pointers.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is now a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is now a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 8-bit unsigned linear interpolated look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points through linear interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_8u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 3 channel 8-bit unsigned linear interpolated look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points through linear interpolation.
        /// NOTE: As of the 5.0 release of NPP, the pValues and pLevels pointers need to be host memory pointers to arrays of device memory pointers.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 3 channel 8-bit unsigned linear interpolated look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points through linear interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_8u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 8-bit unsigned linear interpolated look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points through linear interpolation.
        /// NOTE: As of the 5.0 release of NPP, the pValues and pLevels pointers need to be host memory pointers to arrays of device memory pointers.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 8-bit unsigned linear interpolated look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points through linear interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_8u_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 8-bit unsigned linear interpolated look-up-table color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points through linear interpolation. Alpha channel is the last channel and is not processed.
        /// NOTE: As of the 5.0 release of NPP, the pValues and pLevels pointers need to be host memory pointers to arrays of device memory pointers.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 8-bit unsigned linear interpolated look-up-table in place color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points through linear interpolation. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_8u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 16-bit unsigned look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points using linear interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 16-bit unsigned look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points using linear interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_16u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 3 channel 16-bit unsigned look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points using no interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 3 channel 16-bit unsigned look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points using linear interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_16u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit unsigned look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points using linear interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit unsigned look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points using no interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_16u_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit unsigned look-up-table color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points using no interpolation. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit unsigned look-up-table in place color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points using no interpolation. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_16u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 16-bit signed look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points using linear interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 16-bit signed look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points using linear interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_16s_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 3 channel 16-bit signed look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points using no interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 3 channel 16-bit signed look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points using linear interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_16s_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit signed look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points using linear interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit signed look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points using no interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_16s_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit signed look-up-table color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points using no interpolation. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit signed look-up-table in place color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points using no interpolation. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_16s_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 32-bit floating point look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points using linear interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 32-bit floating point look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points using linear interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_32f_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 3 channel 32-bit floating point look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points using no interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 3 channel 32-bit floating point look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points using linear interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_32f_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 32-bit floating point look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points using linear interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 32-bit floating point look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points using no interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_32f_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 32-bit floating point look-up-table color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points using no interpolation. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 32-bit floating point look-up-table in place color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points using no interpolation. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Linear_32f_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 8-bit unsigned cubic interpolated look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points through cubic interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 8-bit unsigned cubic interpolated look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points through cubic interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_8u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 3 channel 8-bit unsigned cubic interpolated look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points through cubic interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 3 channel 8-bit unsigned cubic interpolated look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points through cubic interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_8u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 8-bit unsigned cubic interpolated look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points through cubic interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 8-bit unsigned cubic interpolated look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points through cubic interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_8u_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 8-bit unsigned cubic interpolated look-up-table color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points through cubic interpolation. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 8-bit unsigned cubic interpolated look-up-table in place color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points through cubic interpolation. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_8u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 16-bit unsigned look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points through cubic interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 16-bit unsigned look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points through cubic interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_16u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 3 channel 16-bit unsigned look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points using no interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 3 channel 16-bit unsigned look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points through cubic interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_16u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit unsigned look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points through cubic interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit unsigned look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points using no interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_16u_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit unsigned look-up-table color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points using no interpolation. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit unsigned look-up-table in place color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points using no interpolation. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_16u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 16-bit signed look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points through cubic interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 16-bit signed look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points through cubic interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_16s_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 3 channel 16-bit signed look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points using no interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 3 channel 16-bit signed look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points through cubic interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_16s_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit signed look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points through cubic interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit signed look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points using no interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_16s_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit signed look-up-table color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points using no interpolation. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 16-bit signed look-up-table in place color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points using no interpolation. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_16s_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 32-bit floating point look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points through cubic interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 32-bit floating point look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points through cubic interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Pointer to an array of user defined OUTPUT values (this is a device memory pointer)</param>
        /// <param name="pLevels">Pointer to an array of user defined INPUT values (this is a device memory pointer)</param>
        /// <param name="nLevels">Number of user defined number of input/output mapping points (levels)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_32f_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            IntPtr pLevels,
            int nLevels);

        /// <summary>
        /// 3 channel 32-bit floating point look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points using no interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 3 channel 32-bit floating point look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points through cubic interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_32f_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 32-bit floating point look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points through cubic interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 32-bit floating point look-up-table in place color conversion.
        /// The LUT is derived from a set of user defined mapping points using no interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 4 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_32f_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] nLevels);

        /// <summary>
        /// 4 channel 32-bit floating point look-up-table color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points using no interpolation. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// 4 channel 32-bit floating point look-up-table in place color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points using no interpolation. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT values.</param>
        /// <param name="pLevels">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined INPUT values.</param>
        /// <param name="nLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per color CHANNEL.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Cubic_32f_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] nLevels);

        /// <summary>
        /// Four channel 8-bit unsigned 3D trilinear interpolated look-up-table color conversion, with alpha copy. Alpha channel is the last channel and is copied to the destination unmodified.
        /// The LUT is derived from a set of user defined mapping points through trilinear interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Device pointer to aLevels[2] number of contiguous 2D x,y planes of 4-byte packed RGBX values containing the user defined base OUTPUT values at that x,y, and z (R,G,B) level location. Each level must contain x * y 4-byte packed pixel values (4th byte is used for alignement only and is ignored) in row (x) order.</param>
        /// <param name="pLevels">Host pointer to an array of 3 host pointers, one per cube edge, pointing to user defined INPUT level values.</param>
        /// <param name="aLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per 3D cube edge. aLevels[0] represents the number of x axis levels (Red), aLevels[1] represents the number of y axis levels (Green), and aLevels[2] represets the number of z axis levels (Blue).</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Trilinear_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] aLevels);

        /// <summary>
        /// Four channel 8-bit unsigned 3D trilinear interpolated look-up-table color conversion, not affecting alpha. Alpha channel is the last channel and is not processed.
        /// The LUT is derived from a set of user defined mapping points through trilinear interpolation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Device pointer to aLevels[2] number of contiguous 2D x,y planes of 4-byte packed RGBX values containing the user defined base OUTPUT values at that x,y, and z (R,G,B) level location. Each level must contain x * y 4-byte packed pixel values (4th byte is used for alignement only and is ignored) in row (x) order.</param>
        /// <param name="pLevels">Host pointer to an array of 3 host pointers, one per cube edge, pointing to user defined INPUT level values.</param>
        /// <param name="aLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per 3D cube edge. aLevels[0] represents the number of x axis levels (Red), aLevels[1] represents the number of y axis levels (Green), and aLevels[2] represets the number of z axis levels (Blue).</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Trilinear_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] aLevels);

        /// <summary>
        /// Four channel 8-bit unsigned 3D trilinear interpolated look-up-table in place color conversion, not affecting alpha. Alpha channel is the last channel and is not processed.
        /// The LUT is derived from a set of user defined mapping points through trilinear interpolation.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pValues">Device pointer aLevels[2] number of contiguous 2D x,y planes of 4-byte packed RGBX values containing the user defined base OUTPUT values at that x,y, and z (R,G,B) level location. Each level must contain x * y 4-byte packed pixel values (4th byte is used for alignement only and is ignored) in row (x) order.</param>
        /// <param name="pLevels">Host pointer to an array of 3 host pointers, one per cube edge, pointing to user defined INPUT level values.</param>
        /// <param name="aLevels">Host pointer to an array of 3 user defined number of input/output mapping points, one per 3D cube edge. aLevels[0] represents the number of x axis levels (Red), aLevels[1] represents the number of y axis levels (Green), and aLevels[2] represets the number of z axis levels (Blue).</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUT_Trilinear_8u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            IntPtr pValues,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pLevels,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] aLevels);

        /// <summary>
        /// One channel 8-bit unsigned bit range restricted palette look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points in a palette and source pixels are then processed using a restricted bit range when looking up palette values.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pTable">Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)</param>
        /// <param name="nBitSize">Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUTPalette_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pTable,
            int nBitSize);

        /// <summary>
        /// One channel 8-bit unsigned bit range restricted 24-bit palette look-up-table color conversion with 24-bit destination output per pixel.
        /// The LUT is derived from a set of user defined mapping points in a palette and source pixels are then processed using a restricted bit range when looking up palette values.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step (3 bytes per pixel).</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pTable">Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)</param>
        /// <param name="nBitSize">Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUTPalette_8u24u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pTable,
            int nBitSize);

        /// <summary>
        /// One channel 8-bit unsigned bit range restricted 32-bit palette look-up-table color conversion with 32-bit destination output per pixel.
        /// The LUT is derived from a set of user defined mapping points in a palette and source pixels are then processed using a restricted bit range when looking up palette values.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step (4 bytes per pixel).</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pTable">Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)</param>
        /// <param name="nBitSize">Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUTPalette_8u32u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pTable,
            int nBitSize);

        /// <summary>
        /// Three channel 8-bit unsigned bit range restricted palette look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points in a palette and source pixels are then processed using a restricted bit range when looking up palette values.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pTables">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.</param>
        /// <param name="nBitSize">Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUTPalette_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pTables,
            int nBitSize);

        /// <summary>
        /// Four channel 8-bit unsigned bit range restricted palette look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points in a palette and source pixels are then processed using a restricted bit range when looking up palette values.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pTables">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.</param>
        /// <param name="nBitSize">Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUTPalette_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pTables,
            int nBitSize);

        /// <summary>
        /// Four channel 8-bit unsigned bit range restricted palette look-up-table color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points in a palette and source pixels are then processed using a restricted bit range when looking up palette values. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pTables">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.</param>
        /// <param name="nBitSize">Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUTPalette_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pTables,
            int nBitSize);

        /// <summary>
        /// One channel 16-bit unsigned bit range restricted palette look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points in a palette and source pixels are then processed using a restricted bit range when looking up palette values.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pTable">Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)</param>
        /// <param name="nBitSize">Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUTPalette_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pTable,
            int nBitSize);

        /// <summary>
        /// One channel 16-bit unsigned bit range restricted 8-bit unsigned palette look-up-table color conversion with 8-bit unsigned destination output per pixel.
        /// The LUT is derived from a set of user defined mapping points in a palette and source pixels are then processed using a restricted bit range when looking up palette values.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step (1 unsigned byte per pixel).</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pTable">Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)</param>
        /// <param name="nBitSize">Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUTPalette_16u8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pTable,
            int nBitSize);

        /// <summary>
        /// One channel 16-bit unsigned bit range restricted 24-bit unsigned palette look-up-table color conversion with 24-bit unsigned destination output per pixel.
        /// The LUT is derived from a set of user defined mapping points in a palette and source pixels are then processed using a restricted bit range when looking up palette values.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step (3 unsigned bytes per pixel).</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pTable">Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)</param>
        /// <param name="nBitSize">Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUTPalette_16u24u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pTable,
            int nBitSize);

        /// <summary>
        /// One channel 16-bit unsigned bit range restricted 32-bit palette look-up-table color conversion with 32-bit unsigned destination output per pixel.
        /// The LUT is derived from a set of user defined mapping points in a palette and source pixels are then processed using a restricted bit range when looking up palette values.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step (4 bytes per pixel).</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pTable">Pointer to an array of user defined OUTPUT palette values (this is a device memory pointer)</param>
        /// <param name="nBitSize">Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUTPalette_16u32u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pTable,
            int nBitSize);

        /// <summary>
        /// Three channel 16-bit unsigned bit range restricted palette look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points in a palette and source pixels are then processed using a restricted bit range when looking up palette values.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pTables">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.</param>
        /// <param name="nBitSize">Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUTPalette_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pTables,
            int nBitSize);

        /// <summary>
        /// Four channel 16-bit unsigned bit range restricted palette look-up-table color conversion.
        /// The LUT is derived from a set of user defined mapping points in a palette and source pixels are then processed using a restricted bit range when looking up palette values.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pTables">Host pointer to an array of 4 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.</param>
        /// <param name="nBitSize">Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUTPalette_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] pTables,
            int nBitSize);

        /// <summary>
        /// Four channel 16-bit unsigned bit range restricted palette look-up-table color conversion, not affecting Alpha.
        /// The LUT is derived from a set of user defined mapping points in a palette and source pixels are then processed using a restricted bit range when looking up palette values. Alpha channel is the last channel and is not processed.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pTables">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values.</param>
        /// <param name="nBitSize">Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUTPalette_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pTables,
            int nBitSize);

        /// <summary>
        /// Three channel 8-bit unsigned source bit range restricted palette look-up-table color conversion to four channel 8-bit unsigned destination output with alpha.
        /// The LUT is derived from a set of user defined mapping points in a palette and source pixels are then processed using a restricted bit range when looking up palette values. This function also reverses the source pixel channel order in the destination so the Alpha channel is the first channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step (3 bytes per pixel).</param>
        /// <param name="nAlphaValue">Signed alpha value that will be used to initialize the pixel alpha channel position in all modified destination pixels.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step (4 bytes per pixel with alpha).</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pTables">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values. Alpha values < 0 or > 255 will cause destination pixel alpha channel values to be unmodified.</param>
        /// <param name="nBitSize">Number of least significant bits (must be > 0 and <= 8) of each source pixel value to use as index into palette table during conversion.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUTPaletteSwap_8u_C3A0C4R(
            IntPtr pSrc,
            int nSrcStep,
            int nAlphaValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pTables,
            int nBitSize);

        /// <summary>
        /// Three channel 16-bit unsigned source bit range restricted palette look-up-table color conversion to four channel 16-bit unsigned destination output with alpha.
        /// The LUT is derived from a set of user defined mapping points in a palette and source pixels are then processed using a restricted bit range when looking up palette values. This function also reverses the source pixel channel order in the destination so the Alpha channel is the first channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step (3 unsigned short integers per pixel).</param>
        /// <param name="nAlphaValue">Signed alpha value that will be used to initialize the pixel alpha channel position in all modified destination pixels.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step (4 unsigned short integers per pixel with alpha).</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pTables">Host pointer to an array of 3 device memory pointers, one per color CHANNEL, pointing to user defined OUTPUT palette values. Alpha values < 0 or > 65535 will cause destination pixel alpha channel values to be unmodified.</param>
        /// <param name="nBitSize">Number of least significant bits (must be > 0 and <= 16) of each source pixel value to use as index into palette table during conversion.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLUTPaletteSwap_16u_C3A0C4R(
            IntPtr pSrc,
            int nSrcStep,
            int nAlphaValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] pTables,
            int nBitSize);

    }
}