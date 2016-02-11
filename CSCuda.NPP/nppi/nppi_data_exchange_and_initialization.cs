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
        /// 8-bit image set.
        /// </summary>
        /// <param name="nValue">The pixel value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_8s_C1R(
            Npp8s nValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 8-bit two-channel image set.
        /// </summary>
        /// <param name="aValue">The pixel value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_8s_C2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]Npp8s[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 8-bit three-channel image set.
        /// </summary>
        /// <param name="aValue">The pixel value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_8s_C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8s[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 8-bit four-channel image set.
        /// </summary>
        /// <param name="aValue">The pixel value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_8s_C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp8s[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 8-bit four-channel image set ignoring alpha channel.
        /// </summary>
        /// <param name="aValue">The pixel value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_8s_AC4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8s[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 8-bit unsigned image set.
        /// </summary>
        /// <param name="nValue">The pixel value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_8u_C1R(
            Npp8u nValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 8-bit unsigned image set.
        /// </summary>
        /// <param name="aValue">The pixel value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_8u_C2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]Npp8u[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 8-bit unsigned image set.
        /// </summary>
        /// <param name="aValue">The pixel value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_8u_C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_8u_C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp8u[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned image set method, not affecting Alpha channel.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_8u_AC4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 16-bit unsigned image set.
        /// </summary>
        /// <param name="nValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16u_C1R(
            Npp16u nValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 16-bit unsigned image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16u_C2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]Npp16u[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 16-bit unsigned image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16u_C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 16-bit unsigned image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16u_C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16u[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 16-bit unsigned image set method, not affecting Alpha channel.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16u_AC4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 16-bit image set.
        /// </summary>
        /// <param name="nValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16s_C1R(
            Npp16s nValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 16-bit image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16s_C2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]Npp16s[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 16-bit image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16s_C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 16-bit image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16s_C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16s[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 16-bit image set method, not affecting Alpha channel.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16s_AC4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 16-bit complex integer image set.
        /// </summary>
        /// <param name="oValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16sc_C1R(
            Npp16sc oValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 16-bit complex integer two-channel image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16sc_C2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]Npp16sc[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 16-bit complex integer three-channel image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16sc_C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16sc[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 16-bit complex integer four-channel image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16sc_C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16sc[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 16-bit complex integer four-channel image set ignoring alpha.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16sc_AC4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16sc[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 32-bit image set.
        /// </summary>
        /// <param name="nValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32s_C1R(
            Npp32s nValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 32-bit image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32s_C2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]Npp32s[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 32-bit image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32s_C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 32-bit image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32s_C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32s[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 32-bit image set method, not affecting Alpha channel.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32s_AC4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 32-bit unsigned image set.
        /// </summary>
        /// <param name="nValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32u_C1R(
            Npp32u nValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 32-bit unsigned image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32u_C2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]Npp32u[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 32-bit unsigned image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32u_C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 32-bit unsigned image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32u_C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32u[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 32-bit unsigned image set method, not affecting Alpha channel.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32u_AC4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 32-bit complex integer image set.
        /// </summary>
        /// <param name="oValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32sc_C1R(
            Npp32sc oValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Two channel 32-bit complex integer image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32sc_C2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]Npp32sc[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 32-bit complex integer image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32sc_C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32sc[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 32-bit complex integer image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32sc_C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32sc[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 32-bit complex integer four-channel image set ignoring alpha.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32sc_AC4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32sc[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 32-bit floating point image set.
        /// </summary>
        /// <param name="nValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32f_C1R(
            Npp32f nValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 2 channel 32-bit floating point image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32f_C2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]Npp32f[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 32-bit floating point image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32f_C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 32-bit floating point image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32f_C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32f[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 32-bit floating point image set method, not affecting Alpha channel.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32f_AC4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 32-bit complex image set.
        /// </summary>
        /// <param name="oValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32fc_C1R(
            Npp32fc oValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Two channel 32-bit complex image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32fc_C2R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 2)]Npp32fc[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 32-bit complex image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32fc_C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32fc[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 32-bit complex image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32fc_C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32fc[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 32-bit complex four-channel image set ignoring alpha.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32fc_AC4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32fc[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Masked 8-bit unsigned image set.
        /// </summary>
        /// <param name="nValue">The pixel value to be set.</param>
        /// <param name="pDst">Pointer destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_8u_C1MR(
            Npp8u nValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// Masked 3 channel 8-bit unsigned image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_8u_C3MR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// Masked 4 channel 8-bit unsigned image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_8u_C4MR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp8u[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// Masked 4 channel 8-bit unsigned image set method, not affecting Alpha channel.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_8u_AC4MR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// Masked 16-bit unsigned image set.
        /// </summary>
        /// <param name="nValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16u_C1MR(
            Npp16u nValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// Masked 3 channel 16-bit unsigned image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16u_C3MR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// Masked 4 channel 16-bit unsigned image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16u_C4MR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16u[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// Masked 4 channel 16-bit unsigned image set method, not affecting Alpha channel.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16u_AC4MR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// Masked 16-bit image set.
        /// </summary>
        /// <param name="nValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16s_C1MR(
            Npp16s nValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// Masked 3 channel 16-bit image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16s_C3MR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// Masked 4 channel 16-bit image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16s_C4MR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16s[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// Masked 4 channel 16-bit image set method, not affecting Alpha channel.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16s_AC4MR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// Masked 32-bit image set.
        /// </summary>
        /// <param name="nValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32s_C1MR(
            Npp32s nValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// Masked 3 channel 32-bit image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32s_C3MR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// Masked 4 channel 32-bit image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32s_C4MR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32s[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// Masked 4 channel 16-bit image set method, not affecting Alpha channel.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32s_AC4MR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// Masked 32-bit floating point image set.
        /// </summary>
        /// <param name="nValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32f_C1MR(
            Npp32f nValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// Masked 3 channel 32-bit floating point image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32f_C3MR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// Masked 4 channel 32-bit floating point image set.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32f_C4MR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32f[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// Masked 4 channel 32-bit floating point image set method, not affecting Alpha channel.
        /// </summary>
        /// <param name="aValue">The pixel-value to be set.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32f_AC4MR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// 3 channel 8-bit unsigned image set affecting only single channel.
        /// </summary>
        /// <param name="nValue">The pixel-value to be set.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_8u_C3CR(
            Npp8u nValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned image set affecting only single channel.
        /// </summary>
        /// <param name="nValue">The pixel-value to be set.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_8u_C4CR(
            Npp8u nValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 16-bit unsigned image set affecting only single channel.
        /// </summary>
        /// <param name="nValue">The pixel-value to be set.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16u_C3CR(
            Npp16u nValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 16-bit unsigned image set affecting only single channel.
        /// </summary>
        /// <param name="nValue">The pixel-value to be set.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16u_C4CR(
            Npp16u nValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 16-bit signed image set affecting only single channel.
        /// </summary>
        /// <param name="nValue">The pixel-value to be set.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16s_C3CR(
            Npp16s nValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 16-bit signed image set affecting only single channel.
        /// </summary>
        /// <param name="nValue">The pixel-value to be set.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_16s_C4CR(
            Npp16s nValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 32-bit unsigned image set affecting only single channel.
        /// </summary>
        /// <param name="nValue">The pixel-value to be set.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32s_C3CR(
            Npp32s nValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 32-bit unsigned image set affecting only single channel.
        /// </summary>
        /// <param name="nValue">The pixel-value to be set.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32s_C4CR(
            Npp32s nValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 3 channel 32-bit floating point image set affecting only single channel.
        /// </summary>
        /// <param name="nValue">The pixel-value to be set.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32f_C3CR(
            Npp32f nValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 32-bit floating point image set affecting only single channel.
        /// </summary>
        /// <param name="nValue">The pixel-value to be set.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSet_32f_C4CR(
            Npp32f nValue,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 8-bit image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_8s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Two-channel 8-bit image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_8s_C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel 8-bit image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_8s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 8-bit image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_8s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 8-bit image copy, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_8s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 8-bit unsigned image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 8-bit unsigned image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 8-bit unsigned image copy, not affecting Alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 16-bit unsigned image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 16-bit unsigned image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 16-bit unsigned image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 16-bit unsigned image copy, not affecting Alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 16-bit image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 16-bit image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 16-bit image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 16-bit image copy, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 16-bit complex image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16sc_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Two-channel 16-bit complex image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16sc_C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel 16-bit complex image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16sc_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 16-bit complex image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16sc_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 16-bit complex image copy, ignoring alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16sc_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 32-bit image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 32-bit image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 32-bit image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 32-bit image copy, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 32-bit complex image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32sc_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Two-channel 32-bit complex image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32sc_C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel 32-bit complex image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32sc_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 32-bit complex image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32sc_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 32-bit complex image copy, ignoring alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32sc_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 32-bit floating point image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 32-bit floating point image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 32-bit floating point image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 4 channel 32-bit floating point image copy, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// 32-bit floating-point complex image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32fc_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Two-channel 32-bit floating-point complex image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32fc_C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel 32-bit floating-point complex image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32fc_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 32-bit floating-point complex image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32fc_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 32-bit floating-point complex image copy, ignoring alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32fc_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// masked_operation 8-bit unsigned image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_8u_C1MR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// masked_operation three channel 8-bit unsigned image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_8u_C3MR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// masked_operation four channel 8-bit unsigned image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_8u_C4MR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// masked_operation four channel 8-bit unsigned image copy, ignoring alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_8u_AC4MR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// masked_operation 16-bit unsigned image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16u_C1MR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// masked_operation three channel 16-bit unsigned image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16u_C3MR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// masked_operation four channel 16-bit unsigned image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16u_C4MR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// masked_operation four channel 16-bit unsigned image copy, ignoring alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16u_AC4MR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// masked_operation 16-bit signed image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16s_C1MR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// masked_operation three channel 16-bit signed image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16s_C3MR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// masked_operation four channel 16-bit signed image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16s_C4MR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// masked_operation four channel 16-bit signed image copy, ignoring alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16s_AC4MR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// masked_operation 32-bit signed image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32s_C1MR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// masked_operation three channel 32-bit signed image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32s_C3MR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// masked_operation four channel 32-bit signed image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32s_C4MR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// masked_operation four channel 32-bit signed image copy, ignoring alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32s_AC4MR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// masked_operation 32-bit float image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32f_C1MR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// masked_operation three channel 32-bit float image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32f_C3MR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// masked_operation four channel 32-bit float image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32f_C4MR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// masked_operation four channel 32-bit float image copy, ignoring alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32f_AC4MR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pMask,
            int nMaskStep);

        /// <summary>
        /// Select-channel 8-bit unsigned image copy for three-channel images.
        /// </summary>
        /// <param name="pSrc">select_source_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_8u_C3CR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Select-channel 8-bit unsigned image copy for four-channel images.
        /// </summary>
        /// <param name="pSrc">select_source_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_8u_C4CR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Select-channel 16-bit signed image copy for three-channel images.
        /// </summary>
        /// <param name="pSrc">select_source_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16s_C3CR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Select-channel 16-bit signed image copy for four-channel images.
        /// </summary>
        /// <param name="pSrc">select_source_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16s_C4CR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Select-channel 16-bit unsigned image copy for three-channel images.
        /// </summary>
        /// <param name="pSrc">select_source_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16u_C3CR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Select-channel 16-bit unsigned image copy for four-channel images.
        /// </summary>
        /// <param name="pSrc">select_source_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16u_C4CR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Select-channel 32-bit signed image copy for three-channel images.
        /// </summary>
        /// <param name="pSrc">select_source_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32s_C3CR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Select-channel 32-bit signed image copy for four-channel images.
        /// </summary>
        /// <param name="pSrc">select_source_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32s_C4CR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Select-channel 32-bit float image copy for three-channel images.
        /// </summary>
        /// <param name="pSrc">select_source_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32f_C3CR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Select-channel 32-bit float image copy for four-channel images.
        /// </summary>
        /// <param name="pSrc">select_source_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32f_C4CR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel to single-channel 8-bit unsigned image copy.
        /// </summary>
        /// <param name="pSrc">select_source_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_8u_C3C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel to single-channel 8-bit unsigned image copy.
        /// </summary>
        /// <param name="pSrc">select_source_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_8u_C4C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel to single-channel 16-bit signed image copy.
        /// </summary>
        /// <param name="pSrc">select_source_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16s_C3C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel to single-channel 16-bit signed image copy.
        /// </summary>
        /// <param name="pSrc">select_source_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16s_C4C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel to single-channel 16-bit unsigned image copy.
        /// </summary>
        /// <param name="pSrc">select_source_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16u_C3C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel to single-channel 16-bit unsigned image copy.
        /// </summary>
        /// <param name="pSrc">select_source_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16u_C4C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel to single-channel 32-bit signed image copy.
        /// </summary>
        /// <param name="pSrc">select_source_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32s_C3C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel to single-channel 32-bit signed image copy.
        /// </summary>
        /// <param name="pSrc">select_source_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32s_C4C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel to single-channel 32-bit float image copy.
        /// </summary>
        /// <param name="pSrc">select_source_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32f_C3C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel to single-channel 32-bit float image copy.
        /// </summary>
        /// <param name="pSrc">select_source_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32f_C4C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single-channel to three-channel 8-bit unsigned image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_8u_C1C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single-channel to four-channel 8-bit unsigned image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_8u_C1C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single-channel to three-channel 16-bit signed image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16s_C1C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single-channel to four-channel 16-bit signed image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16s_C1C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single-channel to three-channel 16-bit unsigned image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16u_C1C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single-channel to four-channel 16-bit unsigned image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16u_C1C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single-channel to three-channel 32-bit signed image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32s_C1C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single-channel to four-channel 32-bit signed image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32s_C1C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single-channel to three-channel 32-bit float image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32f_C1C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single-channel to four-channel 32-bit float image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">select_destination_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32f_C1C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel 8-bit unsigned packed to planar image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="aDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_8u_C3P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] aDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 8-bit unsigned packed to planar image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="aDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_8u_C4P4R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] aDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel 16-bit signed packed to planar image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="aDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16s_C3P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] aDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 16-bit signed packed to planar image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="aDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16s_C4P4R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] aDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel 16-bit unsigned packed to planar image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="aDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16u_C3P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] aDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 16-bit unsigned packed to planar image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="aDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16u_C4P4R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] aDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel 32-bit signed packed to planar image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="aDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32s_C3P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] aDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 32-bit signed packed to planar image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="aDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32s_C4P4R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] aDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel 32-bit float packed to planar image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="aDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32f_C3P3R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] aDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 32-bit float packed to planar image copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="aDst">destination_planar_image_pointer_array.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32f_C4P4R(
            IntPtr pSrc,
            int nSrcStep,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] aDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel 8-bit unsigned planar to packed image copy.
        /// </summary>
        /// <param name="aSrc">Planar source_image_pointer.</param>
        /// <param name="nSrcStep">source_planar_image_pointer_array.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_8u_P3C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] aSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 8-bit unsigned planar to packed image copy.
        /// </summary>
        /// <param name="aSrc">Planar source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_8u_P4C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] aSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel 16-bit unsigned planar to packed image copy.
        /// </summary>
        /// <param name="aSrc">Planar source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16u_P3C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] aSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 16-bit unsigned planar to packed image copy.
        /// </summary>
        /// <param name="aSrc">Planar source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16u_P4C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] aSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel 16-bit signed planar to packed image copy.
        /// </summary>
        /// <param name="aSrc">Planar source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16s_P3C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] aSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 16-bit signed planar to packed image copy.
        /// </summary>
        /// <param name="aSrc">Planar source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_16s_P4C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] aSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel 32-bit signed planar to packed image copy.
        /// </summary>
        /// <param name="aSrc">Planar source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32s_P3C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] aSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 32-bit signed planar to packed image copy.
        /// </summary>
        /// <param name="aSrc">Planar source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32s_P4C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] aSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three-channel 32-bit float planar to packed image copy.
        /// </summary>
        /// <param name="aSrc">Planar source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32f_P3C3R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]IntPtr[] aSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four-channel 32-bit float planar to packed image copy.
        /// </summary>
        /// <param name="aSrc">Planar source_planar_image_pointer_array.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopy_32f_P4C4R(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]IntPtr[] aSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit unsigned to 16-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8u16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 8-bit unsigned to 16-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8u16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned to 16-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8u16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned to 16-bit unsigned conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8u16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit unsigned to 16-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8u16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 8-bit unsigned to 16-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8u16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned to 16-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8u16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned to 16-bit signed conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8u16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit unsigned to 32-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8u32s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 8-bit unsigned to 32-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8u32s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned to 32-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8u32s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned to 32-bit signed conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8u32s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit unsigned to 32-bit floating-point conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8u32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 8-bit unsigned to 32-bit floating-point conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8u32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned to 32-bit floating-point conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8u32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned to 32-bit floating-point conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8u32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit signed to 32-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8s32s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 8-bit signed to 32-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8s32s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit signed to 32-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8s32s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit signed to 32-bit signed conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8s32s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit signed to 32-bit floating-point conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8s32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 8-bit signed to 32-bit floating-point conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8s32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit signed to 32-bit floating-point conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8s32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit signed to 32-bit floating-point conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8s32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 16-bit unsigned to 32-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16u32s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 16-bit unsigned to 32-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16u32s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit unsigned to 32-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16u32s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit unsigned to 32-bit signed conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16u32s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 16-bit unsigned to 32-bit floating-point conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16u32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 16-bit unsigned to 32-bit floating-point conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16u32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit unsigned to 32-bit floating-point conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16u32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit unsigned to 32-bit floating-point conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16u32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 16-bit signed to 32-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16s32s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 16-bit signed to 32-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16s32s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit signed to 32-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16s32s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit signed to 32-bit signed conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16s32s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 16-bit signed to 32-bit floating-point conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16s32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 16-bit signed to 32-bit floating-point conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16s32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit signed to 32-bit floating-point conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16s32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit signed to 32-bit floating-point conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16s32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit signed to 8-bit unsigned conversion with saturation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8s8u_C1Rs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit signed to 16-bit unsigned conversion with saturation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8s16u_C1Rs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit signed to 16-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8s16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit signed to 32-bit unsigned conversion with saturation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8s32u_C1Rs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 16-bit signed to 16-bit unsigned conversion with saturation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16s16u_C1Rs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 16-bit signed to 32-bit unsigned conversion with saturation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16s32u_C1Rs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 16-bit unsigned to 32-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16u32u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 32-bit signed to 32-bit unsigned conversion with saturation.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32s32u_C1Rs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 32-bit signed to 32-bit floating-point conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32s32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 32-bit unsigned to 32-bit floating-point conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32u32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 16-bit unsigned to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16u8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 16-bit unsigned to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16u8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit unsigned to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16u8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit unsigned to 8-bit unsigned conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16u8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 16-bit signed to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16s8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 16-bit signed to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16s8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit signed to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16s8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit signed to 8-bit unsigned conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16s8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 32-bit signed to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32s8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 32-bit signed to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32s8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 32-bit signed to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32s8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 32-bit signed to 8-bit unsigned conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32s8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 32-bit signed to 8-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32s8s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 32-bit signed to 8-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32s8s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 32-bit signed to 8-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32s8s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 32-bit signed to 8-bit signed conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32s8s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit unsigned to 8-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">rounding_mode_parameter.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_8u8s_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// Single channel 16-bit unsigned to 8-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">rounding_mode_parameter.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16u8s_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// Single channel 16-bit signed to 8-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">rounding_mode_parameter.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16s8s_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// Single channel 16-bit unsigned to 16-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">rounding_mode_parameter.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_16u16s_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// Single channel 32-bit unsigned to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">rounding_mode_parameter.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32u8u_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// Single channel 32-bit unsigned to 8-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">rounding_mode_parameter.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32u8s_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// Single channel 32-bit unsigned to 16-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">rounding_mode_parameter.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32u16u_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// Single channel 32-bit unsigned to 16-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">rounding_mode_parameter.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32u16s_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// Single channel 32-bit unsigned to 32-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">rounding_mode_parameter.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32u32s_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// Single channel 32-bit unsigned to 16-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">rounding_mode_parameter.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32s16u_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// Single channel 32-bit unsigned to 16-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">rounding_mode_parameter.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32s16s_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// Single channel 32-bit floating point to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">Flag specifying how fractional float values are rounded to integer values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32f8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode);

        /// <summary>
        /// Three channel 32-bit floating point to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">Flag specifying how fractional float values are rounded to integer values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32f8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode);

        /// <summary>
        /// Four channel 32-bit floating point to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">Flag specifying how fractional float values are rounded to integer values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32f8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode);

        /// <summary>
        /// Four channel 32-bit floating point to 8-bit unsigned conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">Flag specifying how fractional float values are rounded to integer values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32f8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode);

        /// <summary>
        /// Single channel 32-bit floating point to 8-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">Flag specifying how fractional float values are rounded to integer values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32f8s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode);

        /// <summary>
        /// Three channel 32-bit floating point to 8-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">Flag specifying how fractional float values are rounded to integer values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32f8s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode);

        /// <summary>
        /// Four channel 32-bit floating point to 8-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">Flag specifying how fractional float values are rounded to integer values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32f8s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode);

        /// <summary>
        /// Four channel 32-bit floating point to 8-bit signed conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">Flag specifying how fractional float values are rounded to integer values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32f8s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode);

        /// <summary>
        /// Single channel 32-bit floating point to 16-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">Flag specifying how fractional float values are rounded to integer values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32f16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode);

        /// <summary>
        /// Three channel 32-bit floating point to 16-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">Flag specifying how fractional float values are rounded to integer values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32f16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode);

        /// <summary>
        /// Four channel 32-bit floating point to 16-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">Flag specifying how fractional float values are rounded to integer values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32f16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode);

        /// <summary>
        /// Four channel 32-bit floating point to 16-bit unsigned conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">Flag specifying how fractional float values are rounded to integer values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32f16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode);

        /// <summary>
        /// Single channel 32-bit floating point to 16-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">Flag specifying how fractional float values are rounded to integer values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32f16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode);

        /// <summary>
        /// Three channel 32-bit floating point to 16-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">Flag specifying how fractional float values are rounded to integer values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32f16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode);

        /// <summary>
        /// Four channel 32-bit floating point to 16-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">Flag specifying how fractional float values are rounded to integer values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32f16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode);

        /// <summary>
        /// Four channel 32-bit floating point to 16-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">Flag specifying how fractional float values are rounded to integer values.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32f16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode);

        /// <summary>
        /// Single channel 32-bit floating point to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">Flag specifying how fractional float values are rounded to integer values.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32f8u_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// Single channel 32-bit floating point to 8-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">Flag specifying how fractional float values are rounded to integer values.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32f8s_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// Single channel 32-bit floating point to 16-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">Flag specifying how fractional float values are rounded to integer values.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32f16u_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// Single channel 32-bit floating point to 16-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">Flag specifying how fractional float values are rounded to integer values.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32f16s_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// Single channel 32-bit floating point to 32-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">Flag specifying how fractional float values are rounded to integer values.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32f32u_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// Single channel 32-bit floating point to 32-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eRoundMode">Flag specifying how fractional float values are rounded to integer values.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiConvert_32f32s_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// Single channel 8-bit unsigned to 16-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_8u16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 8-bit unsigned to 16-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_8u16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned to 16-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_8u16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned to 16-bit unsigned conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_8u16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit unsigned to 16-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_8u16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 8-bit unsigned to 16-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_8u16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned to 16-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_8u16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned to 16-bit signed conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_8u16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit unsigned to 32-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_8u32s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 8-bit unsigned to 32-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_8u32s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned to 32-bit signed conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_8u32s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned to 32-bit signed conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_8u32s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit unsigned to 32-bit floating-point conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nMin">specifies the minimum saturation value to which every output value will be clamped.</param>
        /// <param name="nMax">specifies the maximum saturation value to which every output value will be clamped.</param>
        /// <returns>image_data_error_codes, roi_error_codes, ::NPP_SCALE_RANGE_ERROR indicates an error condition if nMax <= nMin.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_8u32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nMin,
            Npp32f nMax);

        /// <summary>
        /// Three channel 8-bit unsigned to 32-bit floating-point conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nMin">specifies the minimum saturation value to which every output value will be clamped.</param>
        /// <param name="nMax">specifies the maximum saturation value to which every output value will be clamped.</param>
        /// <returns>image_data_error_codes, roi_error_codes, ::NPP_SCALE_RANGE_ERROR indicates an error condition if nMax <= nMin.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_8u32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nMin,
            Npp32f nMax);

        /// <summary>
        /// Four channel 8-bit unsigned to 32-bit floating-point conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nMin">specifies the minimum saturation value to which every output value will be clamped.</param>
        /// <param name="nMax">specifies the maximum saturation value to which every output value will be clamped.</param>
        /// <returns>image_data_error_codes, roi_error_codes, ::NPP_SCALE_RANGE_ERROR indicates an error condition if nMax <= nMin.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_8u32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nMin,
            Npp32f nMax);

        /// <summary>
        /// Four channel 8-bit unsigned to 32-bit floating-point conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nMin">specifies the minimum saturation value to which every output value will be clamped.</param>
        /// <param name="nMax">specifies the maximum saturation value to which every output value will be clamped.</param>
        /// <returns>image_data_error_codes, roi_error_codes, ::NPP_SCALE_RANGE_ERROR indicates an error condition if nMax <= nMin.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_8u32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nMin,
            Npp32f nMax);

        /// <summary>
        /// Single channel 16-bit unsigned to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="hint">algorithm performance or accuracy selector, currently ignored</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_16u8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppHintAlgorithm hint);

        /// <summary>
        /// Three channel 16-bit unsigned to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="hint">algorithm performance or accuracy selector, currently ignored</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_16u8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppHintAlgorithm hint);

        /// <summary>
        /// Four channel 16-bit unsigned to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="hint">algorithm performance or accuracy selector, currently ignored</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_16u8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppHintAlgorithm hint);

        /// <summary>
        /// Four channel 16-bit unsigned to 8-bit unsigned conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="hint">algorithm performance or accuracy selector, currently ignored</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_16u8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppHintAlgorithm hint);

        /// <summary>
        /// Single channel 16-bit signed to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="hint">algorithm performance or accuracy selector, currently ignored</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_16s8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppHintAlgorithm hint);

        /// <summary>
        /// Three channel 16-bit signed to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="hint">algorithm performance or accuracy selector, currently ignored</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_16s8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppHintAlgorithm hint);

        /// <summary>
        /// Four channel 16-bit signed to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="hint">algorithm performance or accuracy selector, currently ignored</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_16s8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppHintAlgorithm hint);

        /// <summary>
        /// Four channel 16-bit signed to 8-bit unsigned conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="hint">algorithm performance or accuracy selector, currently ignored</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_16s8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppHintAlgorithm hint);

        /// <summary>
        /// Single channel 32-bit signed to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="hint">algorithm performance or accuracy selector, currently ignored</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_32s8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppHintAlgorithm hint);

        /// <summary>
        /// Three channel 32-bit signed to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="hint">algorithm performance or accuracy selector, currently ignored</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_32s8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppHintAlgorithm hint);

        /// <summary>
        /// Four channel 32-bit signed to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="hint">algorithm performance or accuracy selector, currently ignored</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_32s8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppHintAlgorithm hint);

        /// <summary>
        /// Four channel 32-bit signed to 8-bit unsigned conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="hint">algorithm performance or accuracy selector, currently ignored</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_32s8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppHintAlgorithm hint);

        /// <summary>
        /// Single channel 32-bit floating point to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nMin">specifies the minimum saturation value to which every output value will be clamped.</param>
        /// <param name="nMax">specifies the maximum saturation value to which every output value will be clamped.</param>
        /// <returns>image_data_error_codes, roi_error_codes, ::NPP_SCALE_RANGE_ERROR indicates an error condition if nMax <= nMin.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_32f8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nMin,
            Npp32f nMax);

        /// <summary>
        /// Three channel 32-bit floating point to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nMin">specifies the minimum saturation value to which every output value will be clamped.</param>
        /// <param name="nMax">specifies the maximum saturation value to which every output value will be clamped.</param>
        /// <returns>image_data_error_codes, roi_error_codes, ::NPP_SCALE_RANGE_ERROR indicates an error condition if nMax <= nMin.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_32f8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nMin,
            Npp32f nMax);

        /// <summary>
        /// Four channel 32-bit floating point to 8-bit unsigned conversion.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nMin">specifies the minimum saturation value to which every output value will be clamped.</param>
        /// <param name="nMax">specifies the maximum saturation value to which every output value will be clamped.</param>
        /// <returns>image_data_error_codes, roi_error_codes, ::NPP_SCALE_RANGE_ERROR indicates an error condition if nMax <= nMin.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_32f8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nMin,
            Npp32f nMax);

        /// <summary>
        /// Four channel 32-bit floating point to 8-bit unsigned conversion, not affecting Alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nMin">specifies the minimum saturation value to which every output value will be clamped.</param>
        /// <param name="nMax">specifies the maximum saturation value to which every output value will be clamped.</param>
        /// <returns>image_data_error_codes, roi_error_codes, ::NPP_SCALE_RANGE_ERROR indicates an error condition if nMax <= nMin.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiScale_32f8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nMin,
            Npp32f nMax);

        /// <summary>
        /// 1 channel 8-bit unsigned integer image copy with constant border color.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region of pixels.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image (inner part) and constant border color (outer part).</param>
        /// <param name="nTopBorderHeight">Height (in pixels) of the top border. The number of pixel rows at the top of the destination ROI that will be filled with the constant border color. nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.</param>
        /// <param name="nLeftBorderWidth">Width (in pixels) of the left border. The width of the border at the right side of the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
        /// <param name="nValue">The pixel value to be set for border pixels.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyConstBorder_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth,
            Npp8u nValue);

        /// <summary>
        /// 3 channel 8-bit unsigned integer image copy with constant border color. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <param name="aValue">Vector of the RGBA values of the border pixels to be set.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyConstBorder_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aValue);

        /// <summary>
        /// 4 channel 8-bit unsigned integer image copy with constant border color. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <param name="aValue">Vector of the RGBA values of the border pixels to be set.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyConstBorder_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp8u[] aValue);

        /// <summary>
        /// 4 channel 8-bit unsigned integer image copy with constant border color with alpha channel unaffected. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <param name="aValue">Vector of the RGB values of the border pixels. Because this method does not affect the destination image's alpha channel, only three components of the border color are needed.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyConstBorder_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aValue);

        /// <summary>
        /// 1 channel 16-bit unsigned integer image copy with constant border color.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region of pixels.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image (inner part) and constant border color (outer part).</param>
        /// <param name="nTopBorderHeight">Height (in pixels) of the top border. The number of pixel rows at the top of the destination ROI that will be filled with the constant border color. nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.</param>
        /// <param name="nLeftBorderWidth">Width (in pixels) of the left border. The width of the border at the right side of the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
        /// <param name="nValue">The pixel value to be set for border pixels.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyConstBorder_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth,
            Npp16u nValue);

        /// <summary>
        /// 3 channel 16-bit unsigned integer image copy with constant border color. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <param name="aValue">Vector of the RGBA values of the border pixels to be set.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyConstBorder_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aValue);

        /// <summary>
        /// 4 channel 16-bit unsigned integer image copy with constant border color. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <param name="aValue">Vector of the RGBA values of the border pixels to be set.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyConstBorder_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16u[] aValue);

        /// <summary>
        /// 4 channel 16-bit unsigned integer image copy with constant border color with alpha channel unaffected. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <param name="aValue">Vector of the RGB values of the border pixels. Because this method does not affect the destination image's alpha channel, only three components of the border color are needed.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyConstBorder_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aValue);

        /// <summary>
        /// 1 channel 16-bit signed integer image copy with constant border color.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region of pixels.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image (inner part) and constant border color (outer part).</param>
        /// <param name="nTopBorderHeight">Height (in pixels) of the top border. The number of pixel rows at the top of the destination ROI that will be filled with the constant border color. nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.</param>
        /// <param name="nLeftBorderWidth">Width (in pixels) of the left border. The width of the border at the right side of the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
        /// <param name="nValue">The pixel value to be set for border pixels.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyConstBorder_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth,
            Npp16s nValue);

        /// <summary>
        /// 3 channel 16-bit signed integer image copy with constant border color. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <param name="aValue">Vector of the RGBA values of the border pixels to be set.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyConstBorder_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] aValue);

        /// <summary>
        /// 4 channel 16-bit signed integer image copy with constant border color. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <param name="aValue">Vector of the RGBA values of the border pixels to be set.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyConstBorder_16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16s[] aValue);

        /// <summary>
        /// 4 channel 16-bit signed integer image copy with constant border color with alpha channel unaffected. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <param name="aValue">Vector of the RGB values of the border pixels. Because this method does not affect the destination image's alpha channel, only three components of the border color are needed.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyConstBorder_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] aValue);

        /// <summary>
        /// 1 channel 32-bit signed integer image copy with constant border color.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region of pixels.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image (inner part) and constant border color (outer part).</param>
        /// <param name="nTopBorderHeight">Height (in pixels) of the top border. The number of pixel rows at the top of the destination ROI that will be filled with the constant border color. nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.</param>
        /// <param name="nLeftBorderWidth">Width (in pixels) of the left border. The width of the border at the right side of the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
        /// <param name="nValue">The pixel value to be set for border pixels.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyConstBorder_32s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth,
            Npp32s nValue);

        /// <summary>
        /// 3 channel 32-bit signed integer image copy with constant border color. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <param name="aValue">Vector of the RGBA values of the border pixels to be set.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyConstBorder_32s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aValue);

        /// <summary>
        /// 4 channel 32-bit signed integer image copy with constant border color. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <param name="aValue">Vector of the RGBA values of the border pixels to be set.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyConstBorder_32s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32s[] aValue);

        /// <summary>
        /// 4 channel 32-bit signed integer image copy with constant border color with alpha channel unaffected. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <param name="aValue">Vector of the RGB values of the border pixels. Because this method does not affect the destination image's alpha channel, only three components of the border color are needed.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyConstBorder_32s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aValue);

        /// <summary>
        /// 1 channel 32-bit floating point image copy with constant border color.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region of pixels.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image (inner part) and constant border color (outer part).</param>
        /// <param name="nTopBorderHeight">Height (in pixels) of the top border. The number of pixel rows at the top of the destination ROI that will be filled with the constant border color. nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.</param>
        /// <param name="nLeftBorderWidth">Width (in pixels) of the left border. The width of the border at the right side of the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
        /// <param name="nValue">The pixel value to be set for border pixels.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyConstBorder_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth,
            Npp32f nValue);

        /// <summary>
        /// 3 channel 32-bit floating point image copy with constant border color. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <param name="aValue">Vector of the RGBA values of the border pixels to be set.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyConstBorder_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aValue);

        /// <summary>
        /// 4 channel 32-bit floating point image copy with constant border color. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <param name="aValue">Vector of the RGBA values of the border pixels to be set.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyConstBorder_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32f[] aValue);

        /// <summary>
        /// 4 channel 32-bit floating point image copy with constant border color with alpha channel unaffected. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <param name="aValue">Vector of the RGB values of the border pixels. Because this method does not affect the destination image's alpha channel, only three components of the border color are needed.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyConstBorder_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aValue);

        /// <summary>
        /// 1 channel 8-bit unsigned integer image copy with nearest source image pixel color.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region of pixels.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image (inner part) and nearest source image pixel color (outer part).</param>
        /// <param name="nTopBorderHeight">Height (in pixels) of the top border. The number of pixel rows at the top of the destination ROI that will be filled with the nearest source image pixel color. nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.</param>
        /// <param name="nLeftBorderWidth">Width (in pixels) of the left border. The width of the border at the right side of the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyReplicateBorder_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 3 channel 8-bit unsigned integer image copy with nearest source image pixel color. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyReplicateBorder_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 4 channel 8-bit unsigned integer image copy with nearest source image pixel color. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyReplicateBorder_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 4 channel 8-bit unsigned integer image copy with nearest source image pixel color with alpha channel unaffected. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyReplicateBorder_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 1 channel 16-bit unsigned integer image copy with nearest source image pixel color.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region of pixels.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image (inner part) and nearest source image pixel color (outer part).</param>
        /// <param name="nTopBorderHeight">Height (in pixels) of the top border. The number of pixel rows at the top of the destination ROI that will be filled with the nearest source image pixel color. nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.</param>
        /// <param name="nLeftBorderWidth">Width (in pixels) of the left border. The width of the border at the right side of the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyReplicateBorder_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 3 channel 16-bit unsigned integer image copy with nearest source image pixel color. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyReplicateBorder_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 4 channel 16-bit unsigned integer image copy with nearest source image pixel color. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyReplicateBorder_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 4 channel 16-bit unsigned image copy with nearest source image pixel color with alpha channel unaffected. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyReplicateBorder_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 1 channel 16-bit signed integer image copy with nearest source image pixel color.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region of pixels.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image (inner part) and nearest source image pixel color (outer part).</param>
        /// <param name="nTopBorderHeight">Height (in pixels) of the top border. The number of pixel rows at the top of the destination ROI that will be filled with the nearest source image pixel color. nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.</param>
        /// <param name="nLeftBorderWidth">Width (in pixels) of the left border. The width of the border at the right side of the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyReplicateBorder_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 3 channel 16-bit signed integer image copy with nearest source image pixel color. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyReplicateBorder_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 4 channel 16-bit signed integer image copy with nearest source image pixel color. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyReplicateBorder_16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 4 channel 16-bit signed integer image copy with nearest source image pixel color with alpha channel unaffected. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyReplicateBorder_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 1 channel 32-bit signed integer image copy with nearest source image pixel color.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region of pixels.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image (inner part) and nearest source image pixel color (outer part).</param>
        /// <param name="nTopBorderHeight">Height (in pixels) of the top border. The number of pixel rows at the top of the destination ROI that will be filled with the nearest source image pixel color. nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.</param>
        /// <param name="nLeftBorderWidth">Width (in pixels) of the left border. The width of the border at the right side of the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyReplicateBorder_32s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 3 channel 32-bit signed image copy with nearest source image pixel color. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyReplicateBorder_32s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 4 channel 32-bit signed integer image copy with nearest source image pixel color. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyReplicateBorder_32s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 4 channel 32-bit signed integer image copy with nearest source image pixel color with alpha channel unaffected. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyReplicateBorder_32s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 1 channel 32-bit floating point image copy with nearest source image pixel color.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region of pixels.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image (inner part) and nearest source image pixel color (outer part).</param>
        /// <param name="nTopBorderHeight">Height (in pixels) of the top border. The number of pixel rows at the top of the destination ROI that will be filled with the nearest source image pixel color. nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.</param>
        /// <param name="nLeftBorderWidth">Width (in pixels) of the left border. The width of the border at the right side of the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyReplicateBorder_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 3 channel 32-bit floating point image copy with nearest source image pixel color. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyReplicateBorder_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 4 channel 32-bit floating point image copy with nearest source image pixel color. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyReplicateBorder_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 4 channel 32-bit floating point image copy with nearest source image pixel color with alpha channel unaffected. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyReplicateBorder_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 1 channel 8-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region of pixels.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image (inner part) and a border consisting of wrapped replication of the source image pixel colors (outer part).</param>
        /// <param name="nTopBorderHeight">Height (in pixels) of the top border. The number of pixel rows at the top of the destination ROI that will be filled with the wrapped replication of the corresponding column of source image pixels colors. nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.</param>
        /// <param name="nLeftBorderWidth">Width (in pixels) of the left border. The width of the border at the right side of the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyWrapBorder_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 3 channel 8-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyWrapBorder_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 4 channel 8-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyWrapBorder_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 4 channel 8-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyWrapBorder_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 1 channel 16-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region of pixels.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image (inner part) and a border consisting of wrapped replication of the source image pixel colors (outer part).</param>
        /// <param name="nTopBorderHeight">Height (in pixels) of the top border. The number of pixel rows at the top of the destination ROI that will be filled with the wrapped replication of the corresponding column of source image pixels colors. nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.</param>
        /// <param name="nLeftBorderWidth">Width (in pixels) of the left border. The width of the border at the right side of the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyWrapBorder_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 3 channel 16-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyWrapBorder_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 4 channel 16-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyWrapBorder_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 4 channel 16-bit unsigned integer image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyWrapBorder_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 1 channel 16-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region of pixels.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image (inner part) and a border consisting of wrapped replication of the source image pixel colors (outer part).</param>
        /// <param name="nTopBorderHeight">Height (in pixels) of the top border. The number of pixel rows at the top of the destination ROI that will be filled with the wrapped replication of the corresponding column of source image pixels colors. nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.</param>
        /// <param name="nLeftBorderWidth">Width (in pixels) of the left border. The width of the border at the right side of the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyWrapBorder_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 3 channel 16-bit signed integer image copy with the borders wrapped by replication of source image pixel colors. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyWrapBorder_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 4 channel 16-bit signed integer image copy with the borders wrapped by replication of source image pixel colors. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyWrapBorder_16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 4 channel 16-bit signed integer image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyWrapBorder_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 1 channel 32-bit signed integer image copy with the borders wrapped by replication of source image pixel colors.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region of pixels.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image (inner part) and a border consisting of wrapped replication of the source image pixel colors (outer part).</param>
        /// <param name="nTopBorderHeight">Height (in pixels) of the top border. The number of pixel rows at the top of the destination ROI that will be filled with the wrapped replication of the corresponding column of source image pixels colors. nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.</param>
        /// <param name="nLeftBorderWidth">Width (in pixels) of the left border. The width of the border at the right side of the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyWrapBorder_32s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 3 channel 32-bit signed integer image copy with the borders wrapped by replication of source image pixel colors. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyWrapBorder_32s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 4 channel 32-bit signed integer image copy with the borders wrapped by replication of source image pixel colors. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyWrapBorder_32s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 4 channel 32-bit signed integer image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyWrapBorder_32s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 1 channel 32-bit floating point image copy with the borders wrapped by replication of source image pixel colors.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region of pixels.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image (inner part) and a border consisting of wrapped replication of the source image pixel colors (outer part).</param>
        /// <param name="nTopBorderHeight">Height (in pixels) of the top border. The number of pixel rows at the top of the destination ROI that will be filled with the wrapped replication of the corresponding column of source image pixels colors. nBottomBorderHeight = oDstSizeROI.height - nTopBorderHeight - oSrcSizeROI.height.</param>
        /// <param name="nLeftBorderWidth">Width (in pixels) of the left border. The width of the border at the right side of the destination ROI is implicitly defined by the size of the source ROI: nRightBorderWidth = oDstSizeROI.width - nLeftBorderWidth - oSrcSizeROI.width.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyWrapBorder_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 3 channel 32-bit floating point image copy with the borders wrapped by replication of source image pixel colors. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyWrapBorder_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 4 channel 32-bit floating point image copy with the borders wrapped by replication of source image pixel colors. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyWrapBorder_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 1 channel 32-bit floating point image copy with the borders wrapped by replication of source image pixel colors with alpha channel unaffected. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSizeROI">Size of the source region-of-interest.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nTopBorderHeight">Height of top border.</param>
        /// <param name="nLeftBorderWidth">Width of left border.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopyWrapBorder_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSizeROI,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            int nTopBorderHeight,
            int nLeftBorderWidth);

        /// <summary>
        /// 1 channel 8-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image, source image ROI is assumed to be same as destination image ROI.</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopySubpix_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            Npp32f nDx,
            Npp32f nDy);

        /// <summary>
        /// 3 channel 8-bit unsigned integer linearly interpolated source image subpixel coordinate color copy. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopySubpix_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            Npp32f nDx,
            Npp32f nDy);

        /// <summary>
        /// 4 channel 8-bit unsigned integer linearly interpolated source image subpixel coordinate color copy. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopySubpix_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            Npp32f nDx,
            Npp32f nDy);

        /// <summary>
        /// 4 channel 8-bit unsigned integer linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopySubpix_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            Npp32f nDx,
            Npp32f nDy);

        /// <summary>
        /// 1 channel 16-bit unsigned integer linearly interpolated source image subpixel coordinate color copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image, source image ROI is assumed to be same as destination image ROI.</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopySubpix_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            Npp32f nDx,
            Npp32f nDy);

        /// <summary>
        /// 3 channel 16-bit unsigned integer linearly interpolated source image subpixel coordinate color copy. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopySubpix_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            Npp32f nDx,
            Npp32f nDy);

        /// <summary>
        /// 4 channel 16-bit unsigned integer linearly interpolated source image subpixel coordinate color copy. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopySubpix_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            Npp32f nDx,
            Npp32f nDy);

        /// <summary>
        /// 4 channel 16-bit unsigned linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopySubpix_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            Npp32f nDx,
            Npp32f nDy);

        /// <summary>
        /// 1 channel 16-bit signed integer linearly interpolated source image subpixel coordinate color copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image, source image ROI is assumed to be same as destination image ROI.</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopySubpix_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            Npp32f nDx,
            Npp32f nDy);

        /// <summary>
        /// 3 channel 16-bit signed integer linearly interpolated source image subpixel coordinate color copy. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopySubpix_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            Npp32f nDx,
            Npp32f nDy);

        /// <summary>
        /// 4 channel 16-bit signed integer linearly interpolated source image subpixel coordinate color copy. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopySubpix_16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            Npp32f nDx,
            Npp32f nDy);

        /// <summary>
        /// 4 channel 16-bit signed integer linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopySubpix_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            Npp32f nDx,
            Npp32f nDy);

        /// <summary>
        /// 1 channel 32-bit signed integer linearly interpolated source image subpixel coordinate color copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image, source image ROI is assumed to be same as destination image ROI.</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopySubpix_32s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            Npp32f nDx,
            Npp32f nDy);

        /// <summary>
        /// 3 channel 32-bit signed linearly interpolated source image subpixel coordinate color copy. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopySubpix_32s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            Npp32f nDx,
            Npp32f nDy);

        /// <summary>
        /// 4 channel 32-bit signed integer linearly interpolated source image subpixel coordinate color copy. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopySubpix_32s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            Npp32f nDx,
            Npp32f nDy);

        /// <summary>
        /// 4 channel 32-bit signed integer linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopySubpix_32s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            Npp32f nDx,
            Npp32f nDy);

        /// <summary>
        /// 1 channel 32-bit floating point linearly interpolated source image subpixel coordinate color copy.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image, source image ROI is assumed to be same as destination image ROI.</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopySubpix_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            Npp32f nDx,
            Npp32f nDy);

        /// <summary>
        /// 3 channel 32-bit floating point linearly interpolated source image subpixel coordinate color copy. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopySubpix_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            Npp32f nDx,
            Npp32f nDy);

        /// <summary>
        /// 4 channel 32-bit floating point linearly interpolated source image subpixel coordinate color copy. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopySubpix_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            Npp32f nDx,
            Npp32f nDy);

        /// <summary>
        /// 4 channel 32-bit floating point linearly interpolated source image subpixel coordinate color copy with alpha channel unaffected. See
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <param name="nDx">Fractional part of source image X coordinate.</param>
        /// <param name="nDy">Fractional part of source image Y coordinate.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiCopySubpix_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI,
            Npp32f nDx,
            Npp32f nDy);

        /// <summary>
        /// 1 channel 8-bit unsigned integer source image duplicated in all 3 channels of destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image, source image ROI is assumed to be same as destination image ROI.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDup_8u_C1C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI);

        /// <summary>
        /// 1 channel 8-bit unsigned integer source image duplicated in all 4 channels of destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDup_8u_C1C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI);

        /// <summary>
        /// 1 channel 8-bit unsigned integer source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDup_8u_C1AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI);

        /// <summary>
        /// 1 channel 16-bit unsigned integer source image duplicated in all 3 channels of destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image, source image ROI is assumed to be same as destination image ROI.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDup_16u_C1C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI);

        /// <summary>
        /// 1 channel 16-bit unsigned integer source image duplicated in all 4 channels of destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDup_16u_C1C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI);

        /// <summary>
        /// 1 channel 16-bit unsigned integer source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDup_16u_C1AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI);

        /// <summary>
        /// 1 channel 16-bit signed integer source image duplicated in all 3 channels of destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image, source image ROI is assumed to be same as destination image ROI.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDup_16s_C1C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI);

        /// <summary>
        /// 1 channel 16-bit signed integer source image duplicated in all 4 channels of destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDup_16s_C1C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI);

        /// <summary>
        /// 1 channel 16-bit signed integer source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDup_16s_C1AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI);

        /// <summary>
        /// 1 channel 32-bit signed integer source image duplicated in all 3 channels of destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image, source image ROI is assumed to be same as destination image ROI.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDup_32s_C1C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI);

        /// <summary>
        /// 1 channel 32-bit signed integer source image duplicated in all 4 channels of destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDup_32s_C1C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI);

        /// <summary>
        /// 1 channel 32-bit signed integer source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDup_32s_C1AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI);

        /// <summary>
        /// 1 channel 32-bit floating point source image duplicated in all 3 channels of destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size (width, height) of the destination region, i.e. the region that gets filled with data from the source image, source image ROI is assumed to be same as destination image ROI.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDup_32f_C1C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI);

        /// <summary>
        /// 1 channel 32-bit floating point source image duplicated in all 4 channels of destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDup_32f_C1C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI);

        /// <summary>
        /// 1 channel 32-bit floating point source image duplicated in 3 channels of 4 channel destination image with alpha channel unaffected.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oDstSizeROI">Size of the destination region-of-interest.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDup_32f_C1AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oDstSizeROI);

        /// <summary>
        /// 1 channel 8-bit unsigned int image transpose.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">Pointer to the destination ROI.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSrcROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiTranspose_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSrcROI);

        /// <summary>
        /// 3 channel 8-bit unsigned int image transpose.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">Pointer to the destination ROI.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSrcROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiTranspose_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSrcROI);

        /// <summary>
        /// 4 channel 8-bit unsigned int image transpose.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">Pointer to the destination ROI.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSrcROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiTranspose_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSrcROI);

        /// <summary>
        /// 1 channel 16-bit unsigned int image transpose.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">Pointer to the destination ROI.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSrcROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiTranspose_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSrcROI);

        /// <summary>
        /// 3 channel 16-bit unsigned int image transpose.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">Pointer to the destination ROI.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSrcROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiTranspose_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSrcROI);

        /// <summary>
        /// 4 channel 16-bit unsigned int image transpose.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">Pointer to the destination ROI.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSrcROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiTranspose_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSrcROI);

        /// <summary>
        /// 1 channel 16-bit signed int image transpose.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">Pointer to the destination ROI.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSrcROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiTranspose_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSrcROI);

        /// <summary>
        /// 3 channel 16-bit signed int image transpose.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">Pointer to the destination ROI.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSrcROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiTranspose_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSrcROI);

        /// <summary>
        /// 4 channel 16-bit signed int image transpose.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">Pointer to the destination ROI.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSrcROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiTranspose_16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSrcROI);

        /// <summary>
        /// 1 channel 32-bit signed int image transpose.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">Pointer to the destination ROI.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSrcROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiTranspose_32s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSrcROI);

        /// <summary>
        /// 3 channel 32-bit signed int image transpose.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">Pointer to the destination ROI.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSrcROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiTranspose_32s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSrcROI);

        /// <summary>
        /// 4 channel 32-bit signed int image transpose.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">Pointer to the destination ROI.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSrcROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiTranspose_32s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSrcROI);

        /// <summary>
        /// 1 channel 32-bit floating point image transpose.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">Pointer to the destination ROI.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSrcROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiTranspose_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSrcROI);

        /// <summary>
        /// 3 channel 32-bit floating point image transpose.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">Pointer to the destination ROI.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSrcROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiTranspose_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSrcROI);

        /// <summary>
        /// 4 channel 32-bit floating point image transpose.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">Pointer to the destination ROI.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSrcROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiTranspose_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSrcROI);

        /// <summary>
        /// 3 channel 8-bit unsigned integer source image to 3 channel destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] aDstOrder);

        /// <summary>
        /// 3 channel 8-bit unsigned integer in place image.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_8u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] aDstOrder);

        /// <summary>
        /// 4 channel 8-bit unsigned integer source image to 3 channel destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_8u_C4C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] aDstOrder);

        /// <summary>
        /// 4 channel 8-bit unsigned integer source image to 4 channel destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] aDstOrder);

        /// <summary>
        /// 4 channel 8-bit unsigned integer in place image.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_8u_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] aDstOrder);

        /// <summary>
        /// 3 channel 8-bit unsigned integer source image to 4 channel destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR channel order.</param>
        /// <param name="nValue">(V) Single channel constant value that can be replicated in one or more of the 4 destination channels. nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that particular destination channel value unmodified.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_8u_C3C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] aDstOrder,
            Npp8u nValue);

        /// <summary>
        /// 4 channel 8-bit unsigned integer source image to 4 channel destination image with destination alpha channel unaffected.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR channel order. of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] aDstOrder);

        /// <summary>
        /// 3 channel 16-bit unsigned integer source image to 3 channel destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] aDstOrder);

        /// <summary>
        /// 3 channel 16-bit unsigned integer in place image.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_16u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] aDstOrder);

        /// <summary>
        /// 4 channel 16-bit unsigned integer source image to 3 channel destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_16u_C4C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] aDstOrder);

        /// <summary>
        /// 4 channel 16-bit unsigned integer source image to 4 channel destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] aDstOrder);

        /// <summary>
        /// 4 channel 16-bit unsigned integer in place image.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_16u_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] aDstOrder);

        /// <summary>
        /// 3 channel 16-bit unsigned integer source image to 4 channel destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR channel order.</param>
        /// <param name="nValue">(V) Single channel constant value that can be replicated in one or more of the 4 destination channels. nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that particular destination channel value unmodified.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_16u_C3C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] aDstOrder,
            Npp16u nValue);

        /// <summary>
        /// 4 channel 16-bit unsigned integer source image to 4 channel destination image with destination alpha channel unaffected.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] aDstOrder);

        /// <summary>
        /// 3 channel 16-bit signed integer source image to 3 channel destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] aDstOrder);

        /// <summary>
        /// 3 channel 16-bit signed integer in place image.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_16s_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] aDstOrder);

        /// <summary>
        /// 4 channel 16-bit signed integer source image to 3 channel destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_16s_C4C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] aDstOrder);

        /// <summary>
        /// 4 channel 16-bit signed integer source image to 4 channel destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] aDstOrder);

        /// <summary>
        /// 4 channel 16-bit signed integer in place image.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_16s_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] aDstOrder);

        /// <summary>
        /// 3 channel 16-bit signed integer source image to 4 channel destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR channel order.</param>
        /// <param name="nValue">(V) Single channel constant value that can be replicated in one or more of the 4 destination channels. nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that particular destination channel value unmodified.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_16s_C3C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] aDstOrder,
            Npp16s nValue);

        /// <summary>
        /// 4 channel 16-bit signed integer source image to 4 channel destination image with destination alpha channel unaffected.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] aDstOrder);

        /// <summary>
        /// 3 channel 32-bit signed integer source image to 3 channel destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_32s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] aDstOrder);

        /// <summary>
        /// 3 channel 32-bit signed integer in place image.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_32s_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] aDstOrder);

        /// <summary>
        /// 4 channel 32-bit signed integer source image to 3 channel destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_32s_C4C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] aDstOrder);

        /// <summary>
        /// 4 channel 32-bit signed integer source image to 4 channel destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_32s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] aDstOrder);

        /// <summary>
        /// 4 channel 32-bit signed integer in place image.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_32s_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] aDstOrder);

        /// <summary>
        /// 3 channel 32-bit signed integer source image to 4 channel destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR channel order.</param>
        /// <param name="nValue">(V) Single channel constant value that can be replicated in one or more of the 4 destination channels. nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that particular destination channel value unmodified.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_32s_C3C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] aDstOrder,
            Npp32s nValue);

        /// <summary>
        /// 4 channel 32-bit signed integer source image to 4 channel destination image with destination alpha channel unaffected.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_32s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] aDstOrder);

        /// <summary>
        /// 3 channel 32-bit floating point source image to 3 channel destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] aDstOrder);

        /// <summary>
        /// 3 channel 32-bit floating point in place image.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">oSizeROI roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGB image, aDstOrder = [2,1,0] converts this to BGR channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_32f_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] aDstOrder);

        /// <summary>
        /// 4 channel 32-bit floating point source image to 3 channel destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to a 3 channel BGR channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_32f_C4C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] aDstOrder);

        /// <summary>
        /// 4 channel 32-bit floating point source image to 4 channel destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] aDstOrder);

        /// <summary>
        /// 4 channel 32-bit floating point in place image.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an ARGB image, aDstOrder = [3,2,1,0] converts this to BGRA channel order.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_32f_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] aDstOrder);

        /// <summary>
        /// 3 channel 32-bit floating point source image to 4 channel destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGB image, aDstOrder = [3,2,1,0] converts this to VBGR channel order.</param>
        /// <param name="nValue">(V) Single channel constant value that can be replicated in one or more of the 4 destination channels. nValue is either written or not written to a particular channel depending on the aDstOrder entry for that destination channel. An aDstOrder value of 3 will output nValue to that channel, an aDstOrder value greater than 3 will leave that particular destination channel value unmodified.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_32f_C3C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]int[] aDstOrder,
            Npp32f nValue);

        /// <summary>
        /// 4 channel 32-bit floating point source image to 4 channel destination image with destination alpha channel unaffected.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="aDstOrder">Host memory integer array describing how channel values are permutated. The n-th entry of the array contains the number of the channel that is stored in the n-th channel of the output image. E.g. Given an RGBA image, aDstOrder = [2,1,0] converts this to BGRA channel order. In the AC4R case, the alpha channel is always assumed to be channel 3.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSwapChannels_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]int[] aDstOrder);

    }
}