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
        /// One 8-bit unsigned char channel image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_8u_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp8u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 8-bit unsigned char channel in place image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_8u_C1IRSfs(
            Npp8u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 8-bit unsigned char channel image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel..</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_8u_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 8-bit unsigned char channel 8-bit unsigned char in place image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel..</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_8u_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel with unmodified alpha image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel..</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_8u_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel with unmodified alpha in place image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel..</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_8u_AC4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel..</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_8u_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_8u_C4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_16u_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel in place image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_16u_C1IRSfs(
            Npp16u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit unsigned short channel image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_16u_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit unsigned short channel in place image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_16u_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel with unmodified alpha image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_16u_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel with unmodified alpha in place image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_16u_AC4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_16u_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_16u_C4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_16s_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16s nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel in place image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_16s_C1IRSfs(
            Npp16s nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short channel image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_16s_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short channel in place image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_16s_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel with unmodified alpha image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_16s_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel with unmodified alpha in place image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_16s_AC4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_16s_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel in place image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_16s_C4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_16sc_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16sc nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_16sc_C1IRSfs(
            Npp16sc nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_16sc_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16sc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_16sc_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16sc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_16sc_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16sc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_16sc_AC4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16sc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer channel image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32s_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32s nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer channel in place image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32s_C1IRSfs(
            Npp32s nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed integer channel image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32s_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed integer channel in place image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32s_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32sc_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32sc nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32sc_C1IRSfs(
            Npp32sc nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32sc_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32sc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32sc_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32sc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32sc_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32sc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32sc_AC4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32sc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit floating point channel image add constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32f_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32f nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point channel in place image add constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32f_C1IR(
            Npp32f nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point channel image add constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32f_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point channel in place image add constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32f_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel with unmodified alpha image add constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32f_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel with unmodified alpha in place image add constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32f_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel image add constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32f_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32f[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel in place image add constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32f_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32f[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image add constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32fc_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32fc nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image add constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32fc_C1IR(
            Npp32fc nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image add constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32fc_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32fc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image add constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32fc_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32fc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha image add constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32fc_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32fc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha in place image add constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32fc_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32fc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image add constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32fc_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32fc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image add constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddC_32fc_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32fc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_8u_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp8u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 8-bit unsigned char channel in place image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_8u_C1IRSfs(
            Npp8u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 8-bit unsigned char channel image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_8u_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 8-bit unsigned char channel 8-bit unsigned char in place image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_8u_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel with unmodified alpha image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_8u_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel with unmodified alpha in place image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_8u_AC4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_8u_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_8u_C4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_16u_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel in place image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_16u_C1IRSfs(
            Npp16u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit unsigned short channel image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_16u_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit unsigned short channel in place image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_16u_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel with unmodified alpha image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_16u_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel with unmodified alpha in place image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_16u_AC4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_16u_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_16u_C4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_16s_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16s nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel in place image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_16s_C1IRSfs(
            Npp16s nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short channel image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_16s_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short channel in place image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_16s_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel with unmodified alpha image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_16s_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel with unmodified alpha in place image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_16s_AC4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_16s_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel in place image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_16s_C4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_16sc_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16sc nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_16sc_C1IRSfs(
            Npp16sc nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_16sc_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16sc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_16sc_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16sc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_16sc_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16sc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_16sc_AC4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16sc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer channel image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32s_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32s nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer channel in place image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32s_C1IRSfs(
            Npp32s nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed integer channel image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32s_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed integer channel in place image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32s_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32sc_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32sc nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32sc_C1IRSfs(
            Npp32sc nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32sc_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32sc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32sc_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32sc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32sc_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32sc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image multiply by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32sc_AC4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32sc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit floating point channel image multiply by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32f_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32f nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point channel in place image multiply by constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32f_C1IR(
            Npp32f nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point channel image multiply by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32f_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point channel in place image multiply by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32f_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel with unmodified alpha image multiply by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32f_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel with unmodified alpha in place image multiply by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32f_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel image multiply by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32f_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32f[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel in place image multiply by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32f_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32f[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image multiply by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32fc_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32fc nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image multiply by constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32fc_C1IR(
            Npp32fc nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image multiply by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32fc_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32fc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image multiply by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32fc_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32fc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha image multiply by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32fc_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32fc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha in place image multiply by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32fc_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32fc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image multiply by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32fc_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32fc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image multiply by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulC_32fc_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32fc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image multiply by constant and scale by max bit width value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulCScale_8u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp8u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel in place image multiply by constant and scale by max bit width value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulCScale_8u_C1IR(
            Npp8u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channel image multiply by constant and scale by max bit width value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulCScale_8u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channel 8-bit unsigned char in place image multiply by constant and scale by max bit width value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulCScale_8u_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel with unmodified alpha image multiply by constant and scale by max bit width value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulCScale_8u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel with unmodified alpha in place image multiply by constant, scale and scale by max bit width value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulCScale_8u_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel image multiply by constant and scale by max bit width value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulCScale_8u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image multiply by constant and scale by max bit width value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulCScale_8u_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel image multiply by constant and scale by max bit width value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulCScale_16u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel in place image multiply by constant and scale by max bit width value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulCScale_16u_C1IR(
            Npp16u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit unsigned short channel image multiply by constant and scale by max bit width value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulCScale_16u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit unsigned short channel in place image multiply by constant and scale by max bit width value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulCScale_16u_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel with unmodified alpha image multiply by constant and scale by max bit width value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulCScale_16u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel with unmodified alpha in place image multiply by constant and scale by max bit width value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulCScale_16u_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel image multiply by constant and scale by max bit width value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulCScale_16u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image multiply by constant and scale by max bit width value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulCScale_16u_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_8u_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp8u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 8-bit unsigned char channel in place image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_8u_C1IRSfs(
            Npp8u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 8-bit unsigned char channel image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_8u_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 8-bit unsigned char channel 8-bit unsigned char in place image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_8u_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel with unmodified alpha image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_8u_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel with unmodified alpha in place image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_8u_AC4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_8u_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_8u_C4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_16u_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel in place image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_16u_C1IRSfs(
            Npp16u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit unsigned short channel image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_16u_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit unsigned short channel in place image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_16u_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel with unmodified alpha image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_16u_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel with unmodified alpha in place image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_16u_AC4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_16u_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_16u_C4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_16s_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16s nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel in place image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_16s_C1IRSfs(
            Npp16s nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short channel image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_16s_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short channel in place image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_16s_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel with unmodified alpha image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_16s_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel with unmodified alpha in place image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_16s_AC4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_16s_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel in place image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_16s_C4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_16sc_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16sc nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_16sc_C1IRSfs(
            Npp16sc nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_16sc_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16sc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_16sc_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16sc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_16sc_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16sc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_16sc_AC4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16sc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer channel image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32s_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32s nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer channel in place image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32s_C1IRSfs(
            Npp32s nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed integer channel image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32s_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed integer channel in place image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32s_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32sc_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32sc nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32sc_C1IRSfs(
            Npp32sc nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32sc_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32sc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32sc_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32sc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32sc_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32sc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32sc_AC4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32sc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit floating point channel image subtract constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32f_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32f nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point channel in place image subtract constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32f_C1IR(
            Npp32f nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point channel image subtract constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32f_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point channel in place image subtract constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32f_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel with unmodified alpha image subtract constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32f_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel with unmodified alpha in place image subtract constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32f_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel image subtract constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32f_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32f[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel in place image subtract constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32f_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32f[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image subtract constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32fc_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32fc nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image subtract constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32fc_C1IR(
            Npp32fc nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image subtract constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32fc_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32fc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image subtract constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32fc_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32fc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha image subtract constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32fc_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32fc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha in place image subtract constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32fc_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32fc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image subtract constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32fc_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32fc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image subtract constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSubC_32fc_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32fc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_8u_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp8u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 8-bit unsigned char channel in place image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_8u_C1IRSfs(
            Npp8u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 8-bit unsigned char channel image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_8u_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 8-bit unsigned char channel 8-bit unsigned char in place image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_8u_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel with unmodified alpha image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_8u_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel with unmodified alpha in place image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_8u_AC4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_8u_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_8u_C4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_16u_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel in place image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_16u_C1IRSfs(
            Npp16u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit unsigned short channel image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_16u_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit unsigned short channel in place image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_16u_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel with unmodified alpha image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_16u_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel with unmodified alpha in place image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_16u_AC4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_16u_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_16u_C4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_16s_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16s nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel in place image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_16s_C1IRSfs(
            Npp16s nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short channel image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_16s_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short channel in place image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_16s_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel with unmodified alpha image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_16s_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel with unmodified alpha in place image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_16s_AC4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_16s_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel in place image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_16s_C4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_16sc_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16sc nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_16sc_C1IRSfs(
            Npp16sc nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_16sc_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16sc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_16sc_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16sc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_16sc_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16sc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_16sc_AC4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16sc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer channel image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32s_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32s nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer channel in place image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32s_C1IRSfs(
            Npp32s nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed integer channel image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32s_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed integer channel in place image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32s_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32sc_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32sc nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32sc_C1IRSfs(
            Npp32sc nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32sc_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32sc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel in place image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32sc_C3IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32sc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32sc_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32sc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 32-bit signed complex integer (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32sc_AC4IRSfs(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32sc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit floating point channel image divided by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32f_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32f nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point channel in place image divided by constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32f_C1IR(
            Npp32f nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point channel image divided by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32f_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point channel in place image divided by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32f_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel with unmodified alpha image divided by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32f_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel with unmodified alpha in place image divided by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32f_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel image divided by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32f_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32f[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel in place image divided by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32f_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32f[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image divided by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32fc_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32fc nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image divided by constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32fc_C1IR(
            Npp32fc nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image divided by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32fc_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32fc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image divided by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32fc_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32fc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha image divided by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32fc_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32fc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel with unmodified alpha in place image divided by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32fc_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32fc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel image divided by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32fc_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32fc[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit complex floating point (32-bit floating point real, 32-bit floating point imaginary) channel in place image divided by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDivC_32fc_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32fc[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image absolute difference with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbsDiffC_8u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp8u nConstant);

        /// <summary>
        /// One 16-bit unsigned short channel image absolute difference with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbsDiffC_16u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp16u nConstant);

        /// <summary>
        /// One 32-bit floating point channel image absolute difference with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbsDiffC_32f_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nConstant);

        /// <summary>
        /// One 8-bit unsigned char channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_8u_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 8-bit unsigned char channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_8u_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 8-bit unsigned char channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_8u_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 8-bit unsigned char channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_8u_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel with unmodified alpha image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_8u_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel with unmodified alpha in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_8u_AC4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_8u_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_8u_C4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_16u_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_16u_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit unsigned short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_16u_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_16u_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel with unmodified alpha image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_16u_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel with unmodified alpha in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_16u_AC4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_16u_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_16u_C4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_16s_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_16s_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_16s_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_16s_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel with unmodified alpha image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_16s_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel with unmodified alpha in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_16s_AC4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_16s_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_16s_C4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_16sc_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_16sc_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_16sc_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_16sc_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_16sc_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_16sc_AC4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32s_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Note: This function is to be deprecated in future NPP releases, use the function above with a scale factor of 0 instead. 32-bit image add. Add the pixel values of corresponding pixels in the ROI and write them to the output image.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32s_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit signed integer channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32s_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed integer channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32s_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32s_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32sc_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32sc_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32sc_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32sc_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32sc_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image addition, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32sc_AC4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit floating point channel image addition.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32f_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point channel in place image addition.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32f_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point channel image addition.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32f_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point channel in place image addition.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32f_C3IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel with unmodified alpha image addition.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32f_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel with unmodified alpha in place image addition.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32f_AC4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel image addition.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32f_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel in place image addition.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32f_C4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image addition.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32fc_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image addition.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32fc_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image addition.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32fc_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image addition.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32fc_C3IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image addition.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32fc_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image addition.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32fc_AC4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image addition.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32fc_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image addition.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAdd_32fc_C4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image squared then added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddSquare_8u32f_C1IMR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pMask,
            int nMaskStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image squared then added to in place floating point destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddSquare_8u32f_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel image squared then added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddSquare_16u32f_C1IMR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pMask,
            int nMaskStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel image squared then added to in place floating point destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddSquare_16u32f_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point channel image squared then added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddSquare_32f_C1IMR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pMask,
            int nMaskStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point channel image squared then added to in place floating point destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddSquare_32f_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image product added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddProduct_8u32f_C1IMR(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pMask,
            int nMaskStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image product added to in place floating point destination image.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddProduct_8u32f_C1IR(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel image product added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddProduct_16u32f_C1IMR(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pMask,
            int nMaskStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel image product added to in place floating point destination image.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddProduct_16u32f_C1IR(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point channel image product added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddProduct_32f_C1IMR(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pMask,
            int nMaskStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point channel image product added to in place floating point destination image.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddProduct_32f_C1IR(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel alpha weighted image added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nAlpha">Alpha weight to be applied to source image pixels (0.0F to 1.0F)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddWeighted_8u32f_C1IMR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pMask,
            int nMaskStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp32f nAlpha);

        /// <summary>
        /// One 8-bit unsigned char channel alpha weighted image added to in place floating point destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nAlpha">Alpha weight to be applied to source image pixels (0.0F to 1.0F)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddWeighted_8u32f_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp32f nAlpha);

        /// <summary>
        /// One 16-bit unsigned short channel alpha weighted image added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nAlpha">Alpha weight to be applied to source image pixels (0.0F to 1.0F)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddWeighted_16u32f_C1IMR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pMask,
            int nMaskStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp32f nAlpha);

        /// <summary>
        /// One 16-bit unsigned short channel alpha weighted image added to in place floating point destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nAlpha">Alpha weight to be applied to source image pixels (0.0F to 1.0F)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddWeighted_16u32f_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp32f nAlpha);

        /// <summary>
        /// One 32-bit floating point channel alpha weighted image added to in place floating point destination image using filter mask (updates destination when mask is non-zero).
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pMask">mask_image_pointer.</param>
        /// <param name="nMaskStep">mask_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nAlpha">Alpha weight to be applied to source image pixels (0.0F to 1.0F)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddWeighted_32f_C1IMR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pMask,
            int nMaskStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp32f nAlpha);

        /// <summary>
        /// One 32-bit floating point channel alpha weighted image added to in place floating point destination image.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nAlpha">Alpha weight to be applied to source image pixels (0.0F to 1.0F)</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAddWeighted_32f_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp32f nAlpha);

        /// <summary>
        /// One 8-bit unsigned char channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_8u_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 8-bit unsigned char channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_8u_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 8-bit unsigned char channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_8u_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 8-bit unsigned char channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_8u_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel with unmodified alpha image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_8u_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel with unmodified alpha in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_8u_AC4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_8u_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_8u_C4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_16u_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_16u_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit unsigned short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_16u_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_16u_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel with unmodified alpha image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_16u_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel with unmodified alpha in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_16u_AC4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_16u_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_16u_C4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_16s_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_16s_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_16s_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_16s_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel with unmodified alpha image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_16s_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel with unmodified alpha in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_16s_AC4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_16s_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_16s_C4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_16sc_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_16sc_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_16sc_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_16sc_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_16sc_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_16sc_AC4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32s_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Note: This function is to be deprecated in future NPP releases, use the function above with a scale factor of 0 instead. 1 channel 32-bit image multiplication. Multiply corresponding pixels in ROI.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32s_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit signed integer channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32s_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed integer channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32s_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32s_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32sc_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32sc_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32sc_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32sc_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32sc_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image multiplication, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32sc_AC4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit floating point channel image multiplication.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32f_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point channel in place image multiplication.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32f_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point channel image multiplication.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32f_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point channel in place image multiplication.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32f_C3IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel with unmodified alpha image multiplication.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32f_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel with unmodified alpha in place image multiplication.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32f_AC4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel image multiplication.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32f_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel in place image multiplication.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32f_C4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image multiplication.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32fc_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image multiplication.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32fc_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image multiplication.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32fc_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image multiplication.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32fc_C3IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image multiplication.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32fc_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image multiplication.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32fc_AC4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image multiplication.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32fc_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image multiplication.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMul_32fc_C4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image multiplication then scale by maximum value for pixel bit width.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulScale_8u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel in place image multiplication then scale by maximum value for pixel bit width.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulScale_8u_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channel image multiplication then scale by maximum value for pixel bit width.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulScale_8u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel in place image multiplication then scale by maximum value for pixel bit width.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulScale_8u_C3IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel with unmodified alpha image multiplication then scale by maximum value for pixel bit width.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulScale_8u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel with unmodified alpha in place image multiplication then scale by maximum value for pixel bit width.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulScale_8u_AC4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel image multiplication then scale by maximum value for pixel bit width.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulScale_8u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image multiplication then scale by maximum value for pixel bit width.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulScale_8u_C4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel image multiplication then scale by maximum value for pixel bit width.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulScale_16u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel in place image multiplication then scale by maximum value for pixel bit width.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulScale_16u_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit unsigned short channel image multiplication then scale by maximum value for pixel bit width.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulScale_16u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel in place image multiplication then scale by maximum value for pixel bit width.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulScale_16u_C3IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel with unmodified alpha image multiplication then scale by maximum value for pixel bit width.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulScale_16u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel with unmodified alpha in place image multiplication then scale by maximum value for pixel bit width.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulScale_16u_AC4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel image multiplication then scale by maximum value for pixel bit width.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulScale_16u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image multiplication then scale by maximum value for pixel bit width.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiMulScale_16u_C4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_8u_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 8-bit unsigned char channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_8u_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 8-bit unsigned char channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_8u_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 8-bit unsigned char channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_8u_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel with unmodified alpha image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_8u_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel with unmodified alpha in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_8u_AC4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_8u_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_8u_C4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_16u_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_16u_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit unsigned short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_16u_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_16u_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel with unmodified alpha image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_16u_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel with unmodified alpha in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_16u_AC4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_16u_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_16u_C4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_16s_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_16s_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_16s_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_16s_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel with unmodified alpha image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_16s_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel with unmodified alpha in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_16s_AC4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_16s_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_16s_C4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_16sc_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_16sc_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_16sc_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_16sc_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_16sc_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_16sc_AC4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32s_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Note: This function is to be deprecated in future NPP releases, use the function above with a scale factor of 0 instead. 32-bit image subtraction. Subtract pSrc1's pixels from corresponding pixels in pSrc2.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32s_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit signed integer channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32s_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed integer channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32s_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32s_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 32-bit signed integer channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32s_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 32-bit signed integer channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32s_C4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32sc_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32sc_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32sc_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32sc_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32sc_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image subtraction, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32sc_AC4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit floating point channel image subtraction.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32f_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point channel in place image subtraction.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32f_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point channel image subtraction.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32f_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point channel in place image subtraction.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32f_C3IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel with unmodified alpha image subtraction.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32f_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel with unmodified alpha in place image subtraction.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32f_AC4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel image subtraction.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32f_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel in place image subtraction.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32f_C4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image subtraction.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32fc_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image subtraction.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32fc_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image subtraction.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32fc_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image subtraction.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32fc_C3IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image subtraction.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32fc_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image subtraction.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32fc_AC4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image subtraction.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32fc_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image subtraction.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSub_32fc_C4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_8u_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_8u_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_8u_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_8u_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel with unmodified alpha image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_8u_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel with unmodified alpha in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_8u_AC4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_8u_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_8u_C4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_16u_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_16u_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_16u_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_16u_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel with unmodified alpha image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_16u_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel with unmodified alpha in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_16u_AC4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_16u_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_16u_C4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_16s_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_16s_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_16s_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_16s_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel with unmodified alpha image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_16s_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel with unmodified alpha in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_16s_AC4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_16s_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_16s_C4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_16sc_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_16sc_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_16sc_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_16sc_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_16sc_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short complex number (16-bit real, 16-bit imaginary) channel with unmodified alpha in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_16sc_AC4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32s_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Note: This function is to be deprecated in future NPP releases, use the function above with a scale factor of 0 instead. 32-bit image division. Divide pixels in pSrc2 by pSrc1's pixels.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32s_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit signed integer channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32s_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed integer channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32s_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32s_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32sc_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32sc_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32sc_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32sc_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32sc_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 32-bit signed integer complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32sc_AC4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit floating point channel image division.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32f_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point channel in place image division.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32f_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point channel image division.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32f_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point channel in place image division.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32f_C3IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel with unmodified alpha image division.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32f_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel with unmodified alpha in place image division.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32f_AC4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel image division.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32f_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel in place image division.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32f_C4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image division.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32fc_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image division.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32fc_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image division.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32fc_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image division.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32fc_C3IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha image division.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32fc_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel with unmodified alpha in place image division.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32fc_AC4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel image division.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32fc_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point complex number (32-bit real, 32-bit imaginary) channel in place image division.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_32fc_C4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_8u_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// One 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_8u_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// Three 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_8u_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// Three 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_8u_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_8u_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_8u_AC4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_8u_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_8u_C4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_16u_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_16u_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_16u_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_16u_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_16u_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_16u_AC4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_16u_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_16u_C4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_16s_C1RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_16s_C1IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_16s_C3RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_16s_C3IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_16s_AC4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel in place image division with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_16s_AC4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_16s_C4RSfs(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel in place image division, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rndMode">Result Rounding mode to be used (NPP_RND_ZERO, NPP_RND_NEAR, or NP_RND_FINANCIAL)</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiDiv_Round_16s_C4IRSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            NppRoundMode rndMode,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel image absolute value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbs_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit signed short channel in place image absolute value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbs_16s_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit signed short channel image absolute value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbs_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit signed short channel in place image absolute value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbs_16s_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit signed short channel image absolute value with unmodified alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbs_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit signed short channel in place image absolute value with unmodified alpha.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbs_16s_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit signed short channel image absolute value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbs_16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit signed short channel in place image absolute value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbs_16s_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point channel image absolute value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbs_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point channel in place image absolute value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbs_32f_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point channel image absolute value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbs_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point channel in place image absolute value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbs_32f_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel image absolute value with unmodified alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbs_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel in place image absolute value with unmodified alpha.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbs_32f_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel image absolute value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbs_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel in place image absolute value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbs_32f_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel absolute difference of image1 minus image2.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbsDiff_8u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channels absolute difference of image1 minus image2.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbsDiff_8u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channels absolute difference of image1 minus image2.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbsDiff_8u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel absolute difference of image1 minus image2.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbsDiff_16u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point channel absolute difference of image1 minus image2.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAbsDiff_32f_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_8u_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 8-bit unsigned char channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_8u_C1IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 8-bit unsigned char channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_8u_C3RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 8-bit unsigned char channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_8u_C3IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_8u_AC4RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_8u_AC4IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_8u_C4RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_8u_C4IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_16u_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_16u_C1IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit unsigned short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_16u_C3RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit unsigned short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_16u_C3IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_16u_AC4RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_16u_AC4IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_16u_C4RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_16u_C4IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_16s_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_16s_C1IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_16s_C3RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_16s_C3IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_16s_AC4RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel in place image squared with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_16s_AC4IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_16s_C4RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel in place image squared, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_16s_C4IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit floating point channel image squared.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point channel in place image squared.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_32f_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point channel image squared.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point channel in place image squared.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_32f_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel image squared with unmodified alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel in place image squared with unmodified alpha.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_32f_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel image squared.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel in place image squared.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqr_32f_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_8u_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 8-bit unsigned char channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_8u_C1IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 8-bit unsigned char channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_8u_C3RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 8-bit unsigned char channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_8u_C3IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_8u_AC4RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_8u_AC4IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_16u_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_16u_C1IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit unsigned short channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_16u_C3RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit unsigned short channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_16u_C3IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_16u_AC4RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_16u_AC4IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_16s_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_16s_C1IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short channel image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_16s_C3RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short channel in place image square root, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_16s_C3IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_16s_AC4RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Four 16-bit signed short channel in place image square root with unmodified alpha, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_16s_AC4IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit floating point channel image square root.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point channel in place image square root.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_32f_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point channel image square root.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point channel in place image square root.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_32f_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel image square root with unmodified alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel in place image square root with unmodified alpha.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_32f_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel image square root.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit floating point channel in place image square root.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiSqrt_32f_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLn_8u_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 8-bit unsigned char channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLn_8u_C1IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 8-bit unsigned char channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLn_8u_C3RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 8-bit unsigned char channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLn_8u_C3IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLn_16u_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLn_16u_C1IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit unsigned short channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLn_16u_C3RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit unsigned short channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLn_16u_C3IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLn_16s_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLn_16s_C1IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short channel image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLn_16s_C3RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short channel in place image natural logarithm, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLn_16s_C3IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit floating point channel image natural logarithm.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLn_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point channel in place image natural logarithm.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLn_32f_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point channel image natural logarithm.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLn_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point channel in place image natural logarithm.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLn_32f_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiExp_8u_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 8-bit unsigned char channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiExp_8u_C1IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 8-bit unsigned char channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiExp_8u_C3RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 8-bit unsigned char channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiExp_8u_C3IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiExp_16u_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit unsigned short channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiExp_16u_C1IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit unsigned short channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiExp_16u_C3RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit unsigned short channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiExp_16u_C3IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiExp_16s_C1RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 16-bit signed short channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiExp_16s_C1IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short channel image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiExp_16s_C3RSfs(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// Three 16-bit signed short channel in place image exponential, scale by 2^(-nScaleFactor), then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiExp_16s_C3IRSfs(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            int nScaleFactor);

        /// <summary>
        /// One 32-bit floating point channel image exponential.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiExp_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit floating point channel in place image exponential.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiExp_32f_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point channel image exponential.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiExp_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit floating point channel in place image exponential.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiExp_32f_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image logical and with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_8u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp8u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel in place image logical and with constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_8u_C1IR(
            Npp8u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channel image logical and with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_8u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channel in place image logical and with constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_8u_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel image logical and with constant with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_8u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image logical and with constant with unmodified alpha.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_8u_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel image logical and with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_8u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image logical and with constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_8u_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel image logical and with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_16u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel in place image logical and with constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_16u_C1IR(
            Npp16u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit unsigned short channel image logical and with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_16u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit unsigned short channel in place image logical and with constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_16u_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel image logical and with constant with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_16u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image logical and with constant with unmodified alpha.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_16u_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel image logical and with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_16u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image logical and with constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_16u_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit signed integer channel image logical and with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_32s_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32s nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit signed integer channel in place image logical and with constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_32s_C1IR(
            Npp32s nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit signed integer channel image logical and with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_32s_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit signed integer channel in place image logical and with constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_32s_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel image logical and with constant with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_32s_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel in place image logical and with constant with unmodified alpha.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_32s_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel image logical and with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_32s_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel in place image logical and with constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAndC_32s_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image logical or with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_8u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp8u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel in place image logical or with constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_8u_C1IR(
            Npp8u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channel image logical or with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_8u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channel in place image logical or with constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_8u_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel image logical or with constant with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_8u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image logical or with constant with unmodified alpha.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_8u_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel image logical or with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_8u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image logical or with constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_8u_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel image logical or with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_16u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel in place image logical or with constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_16u_C1IR(
            Npp16u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit unsigned short channel image logical or with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_16u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit unsigned short channel in place image logical or with constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_16u_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel image logical or with constant with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_16u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image logical or with constant with unmodified alpha.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_16u_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel image logical or with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_16u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image logical or with constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_16u_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit signed integer channel image logical or with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_32s_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32s nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit signed integer channel in place image logical or with constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_32s_C1IR(
            Npp32s nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit signed integer channel image logical or with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_32s_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit signed integer channel in place image logical or with constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_32s_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel image logical or with constant with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_32s_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel in place image logical or with constant with unmodified alpha.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_32s_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel image logical or with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_32s_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel in place image logical or with constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOrC_32s_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image logical exclusive or with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_8u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp8u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel in place image logical exclusive or with constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_8u_C1IR(
            Npp8u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channel image logical exclusive or with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_8u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channel in place image logical exclusive or with constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_8u_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel image logical exclusive or with constant with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_8u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image logical exclusive or with constant with unmodified alpha.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_8u_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel image logical exclusive or with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_8u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp8u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image logical exclusive or with constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_8u_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp8u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel image logical exclusive or with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_16u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel in place image logical exclusive or with constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_16u_C1IR(
            Npp16u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit unsigned short channel image logical exclusive or with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_16u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit unsigned short channel in place image logical exclusive or with constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_16u_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel image logical exclusive or with constant with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_16u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image logical exclusive or with constant with unmodified alpha.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_16u_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel image logical exclusive or with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_16u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image logical exclusive or with constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_16u_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp16u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit signed integer channel image logical exclusive or with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_32s_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32s nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit signed integer channel in place image logical exclusive or with constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_32s_C1IR(
            Npp32s nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit signed integer channel image logical exclusive or with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_32s_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit signed integer channel in place image logical exclusive or with constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_32s_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel image logical exclusive or with constant with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_32s_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel in place image logical exclusive or with constant with unmodified alpha.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_32s_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel image logical exclusive or with constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_32s_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32s[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel in place image logical exclusive or with constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXorC_32s_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32s[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image right shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_8u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel in place image right shift by constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_8u_C1IR(
            Npp32u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channel image right shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_8u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channel in place image right shift by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_8u_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel image right shift by constant with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_8u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image right shift by constant with unmodified alpha.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_8u_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel image right shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_8u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image right shift by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_8u_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit signed char channel image right shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_8s_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit signed char channel in place image right shift by constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_8s_C1IR(
            Npp32u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit signed char channel image right shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_8s_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit signed char channel in place image right shift by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_8s_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit signed char channel image right shift by constant with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_8s_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit signed char channel in place image right shift by constant with unmodified alpha.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_8s_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit signed char channel image right shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_8s_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit signed char channel in place image right shift by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_8s_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel image right shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_16u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel in place image right shift by constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_16u_C1IR(
            Npp32u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit unsigned short channel image right shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_16u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit unsigned short channel in place image right shift by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_16u_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel image right shift by constant with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_16u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image right shift by constant with unmodified alpha.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_16u_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel image right shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_16u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image right shift by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_16u_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit signed short channel image right shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_16s_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit signed short channel in place image right shift by constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_16s_C1IR(
            Npp32u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit signed short channel image right shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_16s_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit signed short channel in place image right shift by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_16s_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit signed short channel image right shift by constant with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_16s_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit signed short channel in place image right shift by constant with unmodified alpha.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_16s_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit signed short channel image right shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_16s_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit signed short channel in place image right shift by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_16s_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit signed integer channel image right shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_32s_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit signed integer channel in place image right shift by constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_32s_C1IR(
            Npp32u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit signed integer channel image right shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_32s_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit signed integer channel in place image right shift by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_32s_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel image right shift by constant with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_32s_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel in place image right shift by constant with unmodified alpha.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_32s_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel image right shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_32s_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel in place image right shift by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiRShiftC_32s_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image left shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_8u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel in place image left shift by constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_8u_C1IR(
            Npp32u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channel image left shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_8u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channel in place image left shift by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_8u_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel image left shift by constant with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_8u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image left shift by constant with unmodified alpha.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_8u_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel image left shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_8u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image left shift by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_8u_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel image left shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_16u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel in place image left shift by constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_16u_C1IR(
            Npp32u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit unsigned short channel image left shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_16u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit unsigned short channel in place image left shift by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_16u_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel image left shift by constant with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_16u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image left shift by constant with unmodified alpha.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_16u_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel image left shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_16u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image left shift by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_16u_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit signed integer channel image left shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_32s_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit signed integer channel in place image left shift by constant.
        /// </summary>
        /// <param name="nConstant">Constant.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_32s_C1IR(
            Npp32u nConstant,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit signed integer channel image left shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_32s_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit signed integer channel in place image left shift by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_32s_C3IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel image left shift by constant with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_32s_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel in place image left shift by constant with unmodified alpha.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_32s_AC4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel image left shift by constant.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_32s_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32u[] aConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel in place image left shift by constant.
        /// </summary>
        /// <param name="aConstants">fixed size array of constant values, one per channel.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiLShiftC_32s_C4IR(
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 4)]Npp32u[] aConstants,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image logical and.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_8u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel in place image logical and.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_8u_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channel image logical and.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_8u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channel in place image logical and.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_8u_C3IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel image logical and with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_8u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image logical and with unmodified alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_8u_AC4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel image logical and.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_8u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image logical and.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_8u_C4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel image logical and.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_16u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel in place image logical and.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_16u_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit unsigned short channel image logical and.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_16u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit unsigned short channel in place image logical and.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_16u_C3IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel image logical and with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_16u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image logical and with unmodified alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_16u_AC4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel image logical and.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_16u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image logical and.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_16u_C4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit signed integer channel image logical and.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_32s_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit signed integer channel in place image logical and.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_32s_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit signed integer channel image logical and.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_32s_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit signed integer channel in place image logical and.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_32s_C3IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel image logical and with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_32s_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel in place image logical and with unmodified alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_32s_AC4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel image logical and.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_32s_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel in place image logical and.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAnd_32s_C4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image logical or.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_8u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel in place image logical or.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_8u_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channel image logical or.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_8u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channel in place image logical or.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_8u_C3IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel image logical or with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_8u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image logical or with unmodified alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_8u_AC4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel image logical or.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_8u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image logical or.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_8u_C4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel image logical or.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_16u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel in place image logical or.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_16u_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit unsigned short channel image logical or.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_16u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit unsigned short channel in place image logical or.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_16u_C3IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel image logical or with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_16u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image logical or with unmodified alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_16u_AC4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel image logical or.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_16u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image logical or.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_16u_C4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit signed integer channel image logical or.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_32s_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit signed integer channel in place image logical or.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_32s_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit signed integer channel image logical or.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_32s_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit signed integer channel in place image logical or.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_32s_C3IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel image logical or with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_32s_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel in place image logical or with unmodified alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_32s_AC4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel image logical or.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_32s_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel in place image logical or.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiOr_32s_C4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image logical exclusive or.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_8u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel in place image logical exclusive or.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_8u_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channel image logical exclusive or.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_8u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channel in place image logical exclusive or.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_8u_C3IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel image logical exclusive or with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_8u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image logical exclusive or with unmodified alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_8u_AC4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel image logical exclusive or.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_8u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image logical exclusive or.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_8u_C4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel image logical exclusive or.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_16u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel in place image logical exclusive or.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_16u_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit unsigned short channel image logical exclusive or.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_16u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit unsigned short channel in place image logical exclusive or.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_16u_C3IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel image logical exclusive or with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_16u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image logical exclusive or with unmodified alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_16u_AC4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel image logical exclusive or.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_16u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image logical exclusive or.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_16u_C4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit signed integer channel image logical exclusive or.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_32s_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 32-bit signed integer channel in place image logical exclusive or.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_32s_C1IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit signed integer channel image logical exclusive or.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_32s_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 32-bit signed integer channel in place image logical exclusive or.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_32s_C3IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel image logical exclusive or with unmodified alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_32s_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel in place image logical exclusive or with unmodified alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_32s_AC4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel image logical exclusive or.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_32s_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 32-bit signed integer channel in place image logical exclusive or.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiXor_32s_C4IR(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image logical not.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiNot_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel in place image logical not.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiNot_8u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channel image logical not.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiNot_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channel in place image logical not.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiNot_8u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel image logical not with unmodified alpha.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiNot_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image logical not with unmodified alpha.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiNot_8u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel image logical not.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiNot_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image logical not.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiNot_8u_C4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image composition using constant alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="nAlpha2">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaCompC_8u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp8u nAlpha1,
            IntPtr pSrc2,
            int nSrc2Step,
            Npp8u nAlpha2,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// Three 8-bit unsigned char channel image composition using constant alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="nAlpha2">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaCompC_8u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp8u nAlpha1,
            IntPtr pSrc2,
            int nSrc2Step,
            Npp8u nAlpha2,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// Four 8-bit unsigned char channel image composition using constant alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="nAlpha2">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaCompC_8u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp8u nAlpha1,
            IntPtr pSrc2,
            int nSrc2Step,
            Npp8u nAlpha2,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// Four 8-bit unsigned char channel image composition with alpha using constant source alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="nAlpha2">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaCompC_8u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp8u nAlpha1,
            IntPtr pSrc2,
            int nSrc2Step,
            Npp8u nAlpha2,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// One 8-bit signed char channel image composition using constant alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="nAlpha2">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaCompC_8s_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp8s nAlpha1,
            IntPtr pSrc2,
            int nSrc2Step,
            Npp8s nAlpha2,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// One 16-bit unsigned short channel image composition using constant alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="nAlpha2">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaCompC_16u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16u nAlpha1,
            IntPtr pSrc2,
            int nSrc2Step,
            Npp16u nAlpha2,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// Three 16-bit unsigned short channel image composition using constant alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="nAlpha2">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaCompC_16u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16u nAlpha1,
            IntPtr pSrc2,
            int nSrc2Step,
            Npp16u nAlpha2,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// Four 16-bit unsigned short channel image composition using constant alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="nAlpha2">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaCompC_16u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16u nAlpha1,
            IntPtr pSrc2,
            int nSrc2Step,
            Npp16u nAlpha2,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// Four 16-bit unsigned short channel image composition with alpha using constant source alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="nAlpha2">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaCompC_16u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16u nAlpha1,
            IntPtr pSrc2,
            int nSrc2Step,
            Npp16u nAlpha2,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// One 16-bit signed short channel image composition using constant alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="nAlpha2">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaCompC_16s_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16s nAlpha1,
            IntPtr pSrc2,
            int nSrc2Step,
            Npp16s nAlpha2,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// One 32-bit unsigned integer channel image composition using constant alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="nAlpha2">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaCompC_32u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32u nAlpha1,
            IntPtr pSrc2,
            int nSrc2Step,
            Npp32u nAlpha2,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// One 32-bit signed integer channel image composition using constant alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="nAlpha2">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaCompC_32s_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32s nAlpha1,
            IntPtr pSrc2,
            int nSrc2Step,
            Npp32s nAlpha2,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// One 32-bit floating point channel image composition using constant alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nAlpha1">Image alpha opacity (0.0 - 1.0).</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="nAlpha2">Image alpha opacity (0.0 - 1.0).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaCompC_32f_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp32f nAlpha1,
            IntPtr pSrc2,
            int nSrc2Step,
            Npp32f nAlpha2,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// One 8-bit unsigned char channel image premultiplication using constant alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaPremulC_8u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp8u nAlpha1,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel in place image premultiplication using constant alpha.
        /// </summary>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaPremulC_8u_C1IR(
            Npp8u nAlpha1,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channel image premultiplication using constant alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaPremulC_8u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp8u nAlpha1,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 8-bit unsigned char channel in place image premultiplication using constant alpha.
        /// </summary>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaPremulC_8u_C3IR(
            Npp8u nAlpha1,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel image premultiplication using constant alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaPremulC_8u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp8u nAlpha1,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image premultiplication using constant alpha.
        /// </summary>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaPremulC_8u_C4IR(
            Npp8u nAlpha1,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel image premultiplication with alpha using constant alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaPremulC_8u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp8u nAlpha1,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image premultiplication with alpha using constant alpha.
        /// </summary>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaPremulC_8u_AC4IR(
            Npp8u nAlpha1,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel image premultiplication using constant alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaPremulC_16u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16u nAlpha1,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 16-bit unsigned short channel in place image premultiplication using constant alpha.
        /// </summary>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaPremulC_16u_C1IR(
            Npp16u nAlpha1,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit unsigned short channel image premultiplication using constant alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaPremulC_16u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16u nAlpha1,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three 16-bit unsigned short channel in place image premultiplication using constant alpha.
        /// </summary>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaPremulC_16u_C3IR(
            Npp16u nAlpha1,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel image premultiplication using constant alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaPremulC_16u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16u nAlpha1,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image premultiplication using constant alpha.
        /// </summary>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaPremulC_16u_C4IR(
            Npp16u nAlpha1,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel image premultiplication with alpha using constant alpha.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaPremulC_16u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            Npp16u nAlpha1,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image premultiplication with alpha using constant alpha.
        /// </summary>
        /// <param name="nAlpha1">Image alpha opacity (0 - max channel pixel value).</param>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaPremulC_16u_AC4IR(
            Npp16u nAlpha1,
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// One 8-bit unsigned char channel image composition using image alpha values (0 - max channel pixel value).
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaComp_8u_AC1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// Four 8-bit unsigned char channel image composition using image alpha values (0 - max channel pixel value).
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaComp_8u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// One 8-bit signed char channel image composition using image alpha values (0 - max channel pixel value).
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaComp_8s_AC1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// One 16-bit unsigned short channel image composition using image alpha values (0 - max channel pixel value).
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaComp_16u_AC1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// Four 16-bit unsigned short channel image composition using image alpha values (0 - max channel pixel value).
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaComp_16u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// One 16-bit signed short channel image composition using image alpha values (0 - max channel pixel value).
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaComp_16s_AC1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// One 32-bit unsigned integer channel image composition using image alpha values (0 - max channel pixel value).
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaComp_32u_AC1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// Four 32-bit unsigned integer channel image composition using image alpha values (0 - max channel pixel value).
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaComp_32u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// One 32-bit signed integer channel image composition using image alpha values (0 - max channel pixel value).
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaComp_32s_AC1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// Four 32-bit signed integer channel image composition using image alpha values (0 - max channel pixel value).
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaComp_32s_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// One 32-bit floating point channel image composition using image alpha values (0.0 - 1.0).
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaComp_32f_AC1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// Four 32-bit floating point channel image composition using image alpha values (0.0 - 1.0).
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eAlphaOp">alpha-blending operation..</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaComp_32f_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppiAlphaOp eAlphaOp);

        /// <summary>
        /// Four 8-bit unsigned char channel image premultiplication with pixel alpha (0 - max channel pixel value).
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaPremul_8u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 8-bit unsigned char channel in place image premultiplication with pixel alpha (0 - max channel pixel value).
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaPremul_8u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel image premultiplication with pixel alpha (0 - max channel pixel value).
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaPremul_16u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four 16-bit unsigned short channel in place image premultiplication with pixel alpha (0 - max channel pixel value).
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppiAlphaPremul_16u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI);

    }
}