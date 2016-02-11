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
        /// 8-bit unsigned char in place signal add constant, scale, then clamp to saturated value
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be added to each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddC_8u_ISfs(
            Npp8u nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 8-bit unsigned charvector add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be added to each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddC_8u_Sfs(
            IntPtr pSrc,
            Npp8u nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short in place signal add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be added to each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddC_16u_ISfs(
            Npp16u nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short vector add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be added to each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddC_16u_Sfs(
            IntPtr pSrc,
            Npp16u nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short in place signal add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be added to each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddC_16s_ISfs(
            Npp16s nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short signal add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be added to each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddC_16s_Sfs(
            IntPtr pSrc,
            Npp16s nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit integer complex number (16 bit real, 16 bit imaginary)signal add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be added to each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddC_16sc_ISfs(
            Npp16sc nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit integer complex number (16 bit real, 16 bit imaginary) signal add constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be added to each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddC_16sc_Sfs(
            IntPtr pSrc,
            Npp16sc nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer in place signal add constant and scale.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be added to each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddC_32s_ISfs(
            Npp32s nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integersignal add constant and scale.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be added to each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddC_32s_Sfs(
            IntPtr pSrc,
            Npp32s nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal add constant and scale.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be added to each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddC_32sc_ISfs(
            Npp32sc nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit integer complex number (32 bit real, 32 bit imaginary) signal add constant and scale.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be added to each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddC_32sc_Sfs(
            IntPtr pSrc,
            Npp32sc nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit floating point in place signal add constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be added to each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddC_32f_I(
            Npp32f nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point signal add constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be added to each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddC_32f(
            IntPtr pSrc,
            Npp32f nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point complex number (32 bit real, 32 bit imaginary) in place signal add constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be added to each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddC_32fc_I(
            Npp32fc nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal add constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be added to each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddC_32fc(
            IntPtr pSrc,
            Npp32fc nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point, in place signal add constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be added to each vector element</param>
        /// <param name="nLength">Length of the vectors, number of items.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddC_64f_I(
            Npp64f nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit floating pointsignal add constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be added to each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddC_64f(
            IntPtr pSrc,
            Npp64f nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point complex number (64 bit real, 64 bit imaginary) in place signal add constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be added to each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddC_64fc_I(
            Npp64fc nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal add constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be added to each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddC_64fc(
            IntPtr pSrc,
            Npp64fc nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point signal add product of signal times constant to destination signal.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddProductC_32f(
            IntPtr pSrc,
            Npp32f nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char in place signal times constant, scale, then clamp to saturated value
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMulC_8u_ISfs(
            Npp8u nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 8-bit unsigned char signal times constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMulC_8u_Sfs(
            IntPtr pSrc,
            Npp8u nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short in place signal times constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMulC_16u_ISfs(
            Npp16u nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short signal times constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMulC_16u_Sfs(
            IntPtr pSrc,
            Npp16u nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short in place signal times constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMulC_16s_ISfs(
            Npp16s nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short signal times constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMulC_16s_Sfs(
            IntPtr pSrc,
            Npp16s nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit integer complex number (16 bit real, 16 bit imaginary)signal times constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMulC_16sc_ISfs(
            Npp16sc nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit integer complex number (16 bit real, 16 bit imaginary)signal times constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMulC_16sc_Sfs(
            IntPtr pSrc,
            Npp16sc nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer in place signal times constant and scale.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMulC_32s_ISfs(
            Npp32s nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer signal times constant and scale.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMulC_32s_Sfs(
            IntPtr pSrc,
            Npp32s nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal times constant and scale.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMulC_32sc_ISfs(
            Npp32sc nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit integer complex number (32 bit real, 32 bit imaginary) signal times constant and scale.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMulC_32sc_Sfs(
            IntPtr pSrc,
            Npp32sc nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit floating point in place signal times constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMulC_32f_I(
            Npp32f nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point signal times constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMulC_32f(
            IntPtr pSrc,
            Npp32f nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point signal times constant with output converted to 16-bit signed integer.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMulC_Low_32f16s(
            IntPtr pSrc,
            Npp32f nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point signal times constant with output converted to 16-bit signed integer with scaling and saturation of output result.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMulC_32f16s_Sfs(
            IntPtr pSrc,
            Npp32f nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit floating point complex number (32 bit real, 32 bit imaginary) in place signal times constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMulC_32fc_I(
            Npp32fc nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal times constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMulC_32fc(
            IntPtr pSrc,
            Npp32fc nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point, in place signal times constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="nLength">Length of the vectors, number of items.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMulC_64f_I(
            Npp64f nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point signal times constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMulC_64f(
            IntPtr pSrc,
            Npp64f nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point signal times constant with in place conversion to 64-bit signed integer and with scaling and saturation of output result.
        /// </summary>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMulC_64f64s_ISfs(
            Npp64f nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 64-bit floating point complex number (64 bit real, 64 bit imaginary) in place signal times constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMulC_64fc_I(
            Npp64fc nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal times constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be multiplied by each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMulC_64fc(
            IntPtr pSrc,
            Npp64fc nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char in place signal subtract constant, scale, then clamp to saturated value
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be subtracted from each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubC_8u_ISfs(
            Npp8u nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 8-bit unsigned char signal subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be subtracted from each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubC_8u_Sfs(
            IntPtr pSrc,
            Npp8u nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short in place signal subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be subtracted from each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubC_16u_ISfs(
            Npp16u nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short signal subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be subtracted from each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubC_16u_Sfs(
            IntPtr pSrc,
            Npp16u nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short in place signal subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be subtracted from each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubC_16s_ISfs(
            Npp16s nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short signal subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be subtracted from each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubC_16s_Sfs(
            IntPtr pSrc,
            Npp16s nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be subtracted from each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubC_16sc_ISfs(
            Npp16sc nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be subtracted from each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubC_16sc_Sfs(
            IntPtr pSrc,
            Npp16sc nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer in place signal subtract constant and scale.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be subtracted from each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubC_32s_ISfs(
            Npp32s nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer signal subtract constant and scale.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be subtracted from each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubC_32s_Sfs(
            IntPtr pSrc,
            Npp32s nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal subtract constant and scale.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be subtracted from each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubC_32sc_ISfs(
            Npp32sc nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit integer complex number (32 bit real, 32 bit imaginary)signal subtract constant and scale.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be subtracted from each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubC_32sc_Sfs(
            IntPtr pSrc,
            Npp32sc nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit floating point in place signal subtract constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be subtracted from each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubC_32f_I(
            Npp32f nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point signal subtract constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be subtracted from each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubC_32f(
            IntPtr pSrc,
            Npp32f nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point complex number (32 bit real, 32 bit imaginary) in place signal subtract constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be subtracted from each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubC_32fc_I(
            Npp32fc nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal subtract constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be subtracted from each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubC_32fc(
            IntPtr pSrc,
            Npp32fc nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point, in place signal subtract constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be subtracted from each vector element</param>
        /// <param name="nLength">Length of the vectors, number of items.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubC_64f_I(
            Npp64f nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point signal subtract constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be subtracted from each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubC_64f(
            IntPtr pSrc,
            Npp64f nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point complex number (64 bit real, 64 bit imaginary) in place signal subtract constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be subtracted from each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubC_64fc_I(
            Npp64fc nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal subtract constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be subtracted from each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubC_64fc(
            IntPtr pSrc,
            Npp64fc nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char in place signal subtract from constant, scale, then clamp to saturated value
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value each vector element is to be subtracted from</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubCRev_8u_ISfs(
            Npp8u nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 8-bit unsigned char signal subtract from constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value each vector element is to be subtracted from</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubCRev_8u_Sfs(
            IntPtr pSrc,
            Npp8u nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short in place signal subtract from constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value each vector element is to be subtracted from</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubCRev_16u_ISfs(
            Npp16u nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short signal subtract from constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value each vector element is to be subtracted from</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubCRev_16u_Sfs(
            IntPtr pSrc,
            Npp16u nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short in place signal subtract from constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value each vector element is to be subtracted from</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubCRev_16s_ISfs(
            Npp16s nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short signal subtract from constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value each vector element is to be subtracted from</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubCRev_16s_Sfs(
            IntPtr pSrc,
            Npp16s nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract from constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value each vector element is to be subtracted from</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubCRev_16sc_ISfs(
            Npp16sc nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit integer complex number (16 bit real, 16 bit imaginary) signal subtract from constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value each vector element is to be subtracted from</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubCRev_16sc_Sfs(
            IntPtr pSrc,
            Npp16sc nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer in place signal subtract from constant and scale.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value each vector element is to be subtracted from</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubCRev_32s_ISfs(
            Npp32s nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integersignal subtract from constant and scale.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value each vector element is to be subtracted from</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubCRev_32s_Sfs(
            IntPtr pSrc,
            Npp32s nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit integer complex number (32 bit real, 32 bit imaginary) in place signal subtract from constant and scale.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value each vector element is to be subtracted from</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubCRev_32sc_ISfs(
            Npp32sc nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit integer complex number (32 bit real, 32 bit imaginary) signal subtract from constant and scale.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value each vector element is to be subtracted from</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubCRev_32sc_Sfs(
            IntPtr pSrc,
            Npp32sc nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit floating point in place signal subtract from constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value each vector element is to be subtracted from</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubCRev_32f_I(
            Npp32f nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point signal subtract from constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value each vector element is to be subtracted from</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubCRev_32f(
            IntPtr pSrc,
            Npp32f nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point complex number (32 bit real, 32 bit imaginary) in place signal subtract from constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value each vector element is to be subtracted from</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubCRev_32fc_I(
            Npp32fc nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal subtract from constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value each vector element is to be subtracted from</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubCRev_32fc(
            IntPtr pSrc,
            Npp32fc nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point, in place signal subtract from constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value each vector element is to be subtracted from</param>
        /// <param name="nLength">Length of the vectors, number of items.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubCRev_64f_I(
            Npp64f nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point signal subtract from constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value each vector element is to be subtracted from</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubCRev_64f(
            IntPtr pSrc,
            Npp64f nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point complex number (64 bit real, 64 bit imaginary) in place signal subtract from constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value each vector element is to be subtracted from</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubCRev_64fc_I(
            Npp64fc nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal subtract from constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value each vector element is to be subtracted from</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSubCRev_64fc(
            IntPtr pSrc,
            Npp64fc nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char in place signal divided by constant, scale, then clamp to saturated value
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be divided into each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDivC_8u_ISfs(
            Npp8u nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 8-bit unsigned char signal divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be divided into each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDivC_8u_Sfs(
            IntPtr pSrc,
            Npp8u nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short in place signal divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be divided into each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDivC_16u_ISfs(
            Npp16u nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short signal divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be divided into each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDivC_16u_Sfs(
            IntPtr pSrc,
            Npp16u nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short in place signal divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be divided into each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDivC_16s_ISfs(
            Npp16s nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short signal divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be divided into each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDivC_16s_Sfs(
            IntPtr pSrc,
            Npp16s nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit integer complex number (16 bit real, 16 bit imaginary)signal divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be divided into each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDivC_16sc_ISfs(
            Npp16sc nValue,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit integer complex number (16 bit real, 16 bit imaginary) signal divided by constant, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be divided into each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDivC_16sc_Sfs(
            IntPtr pSrc,
            Npp16sc nValue,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit floating point in place signal divided by constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be divided into each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDivC_32f_I(
            Npp32f nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point signal divided by constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be divided into each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDivC_32f(
            IntPtr pSrc,
            Npp32f nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point complex number (32 bit real, 32 bit imaginary) in place signal divided by constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be divided into each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDivC_32fc_I(
            Npp32fc nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point complex number (32 bit real, 32 bit imaginary) signal divided by constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be divided into each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDivC_32fc(
            IntPtr pSrc,
            Npp32fc nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point in place signal divided by constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be divided into each vector element</param>
        /// <param name="nLength">Length of the vectors, number of items.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDivC_64f_I(
            Npp64f nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point signal divided by constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be divided into each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDivC_64f(
            IntPtr pSrc,
            Npp64f nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point complex number (64 bit real, 64 bit imaginary) in place signal divided by constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be divided into each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDivC_64fc_I(
            Npp64fc nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point complex number (64 bit real, 64 bit imaginary) signal divided by constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be divided into each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDivC_64fc(
            IntPtr pSrc,
            Npp64fc nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned short in place constant divided by signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be divided by each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDivCRev_16u_I(
            Npp16u nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned short signal divided by constant, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be divided by each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDivCRev_16u(
            IntPtr pSrc,
            Npp16u nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point in place constant divided by signal.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be divided by each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDivCRev_32f_I(
            Npp32f nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point constant divided by signal.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be divided by each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDivCRev_32f(
            IntPtr pSrc,
            Npp32f nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit signed short signal add signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be added to signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_16s(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned short signal add signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be added to signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_16u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit unsigned int signal add signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be added to signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_32u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point signal add signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be added to signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_32f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point signal add signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be added to signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_64f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit complex floating point signal add signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be added to signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_32fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit complex floating point signal add signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be added to signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_64fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char signal add signal with 16-bit unsigned result, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be added to signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_8u16u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit signed short signal add signal with 32-bit floating point result, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be added to signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_16s32f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char add signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal2 elements to be added to signal1 elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_8u_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short add signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal2 elements to be added to signal1 elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_16u_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short add signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal2 elements to be added to signal1 elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_16s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer add signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal2 elements to be added to signal1 elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_32s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 64-bit signed integer add signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal2 elements to be added to signal1 elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_64s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed complex short add signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal2 elements to be added to signal1 elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_16sc_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed complex integer add signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal2 elements to be added to signal1 elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_32sc_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short in place signal add signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be added to signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_16s_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point in place signal add signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be added to signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_32f_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point in place signal add signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be added to signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_64f_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit complex floating point in place signal add signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be added to signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_32fc_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit complex floating point in place signal add signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be added to signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_64fc_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 16/32-bit signed short in place signal add signal with 32-bit signed integer results, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be added to signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_16s32s_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char in place signal add signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be added to signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_8u_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short in place signal add signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be added to signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_16u_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short in place signal add signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be added to signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_16s_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer in place signal add signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be added to signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_32s_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit complex signed short in place signal add signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be added to signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_16sc_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit complex signed integer in place signal add signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be added to signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAdd_32sc_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit floating point signal add product of source signal times destination signal to destination signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddProduct_32f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point signal add product of source signal times destination signal to destination signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddProduct_64f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit complex floating point signal add product of source signal times destination signal to destination signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddProduct_32fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit complex floating point signal add product of source signal times destination signal to destination signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddProduct_64fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit signed short signal add product of source signal1 times source signal2 to destination signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddProduct_16s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed short signal add product of source signal1 times source signal2 to destination signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddProduct_32s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short signal add product of source signal1 times source signal2 to 32-bit signed integer destination signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer. product of source1 and source2 signal elements to be added to destination elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAddProduct_16s32s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short signal times signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be multiplied by signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_16s(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point signal times signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be multiplied by signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_32f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point signal times signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be multiplied by signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_64f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit complex floating point signal times signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be multiplied by signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_32fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit complex floating point signal times signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be multiplied by signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_64fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char signal times signal with 16-bit unsigned result, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be multiplied by signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_8u16u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit signed short signal times signal with 32-bit floating point result, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be multiplied by signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_16s32f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point signal times 32-bit complex floating point signal with complex 32-bit floating point result, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be multiplied by signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_32f32fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char signal times signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal2 elements to be multiplied by signal1 elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_8u_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short signal time signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal2 elements to be multiplied by signal1 elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_16u_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short signal times signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal2 elements to be multiplied by signal1 elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_16s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer signal times signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal2 elements to be multiplied by signal1 elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_32s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed complex short signal times signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal2 elements to be multiplied by signal1 elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_16sc_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed complex integer signal times signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal2 elements to be multiplied by signal1 elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_32sc_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short signal times 16-bit signed short signal, scale, then clamp to 16-bit signed saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal2 elements to be multiplied by signal1 elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_16u16s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short signal times signal, scale, then clamp to 32-bit signed saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal2 elements to be multiplied by signal1 elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_16s32s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer signal times 32-bit complex signed integer signal, scale, then clamp to 32-bit complex integer saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal2 elements to be multiplied by signal1 elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_32s32sc_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer signal times signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal2 elements to be multiplied by signal1 elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_Low_32s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short in place signal times signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_16s_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point in place signal times signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_32f_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point in place signal times signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_64f_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit complex floating point in place signal times signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_32fc_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit complex floating point in place signal times signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_64fc_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit complex floating point in place signal times 32-bit floating point signal, then clamp to 32-bit complex floating point saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_32f32fc_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char in place signal times signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_8u_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short in place signal times signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_16u_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short in place signal times signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_16s_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer in place signal times signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_32s_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit complex signed short in place signal times signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_16sc_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit complex signed integer in place signal times signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_32sc_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit complex signed integer in place signal times 32-bit signed integer signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be multiplied by signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMul_32s32sc_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short signal subtract signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal1 elements to be subtracted from signal2 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSub_16s(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point signal subtract signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal1 elements to be subtracted from signal2 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSub_32f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point signal subtract signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal1 elements to be subtracted from signal2 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSub_64f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit complex floating point signal subtract signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal1 elements to be subtracted from signal2 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSub_32fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit complex floating point signal subtract signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal1 elements to be subtracted from signal2 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSub_64fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit signed short signal subtract 16-bit signed short signal, then clamp and convert to 32-bit floating point saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal1 elements to be subtracted from signal2 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSub_16s32f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char signal subtract signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal1 elements to be subtracted from signal2 elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSub_8u_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short signal subtract signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal1 elements to be subtracted from signal2 elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSub_16u_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short signal subtract signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal1 elements to be subtracted from signal2 elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSub_16s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer signal subtract signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal1 elements to be subtracted from signal2 elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSub_32s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed complex short signal subtract signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal1 elements to be subtracted from signal2 elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSub_16sc_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed complex integer signal subtract signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal1 elements to be subtracted from signal2 elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSub_32sc_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short in place signal subtract signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSub_16s_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point in place signal subtract signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSub_32f_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point in place signal subtract signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSub_64f_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit complex floating point in place signal subtract signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSub_32fc_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit complex floating point in place signal subtract signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSub_64fc_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char in place signal subtract signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSub_8u_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short in place signal subtract signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSub_16u_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short in place signal subtract signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSub_16s_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer in place signal subtract signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSub_32s_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit complex signed short in place signal subtract signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSub_16sc_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit complex signed integer in place signal subtract signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal1 elements to be subtracted from signal2 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSub_32sc_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 8-bit unsigned char signal divide signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_8u_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short signal divide signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_16u_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short signal divide signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_16s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer signal divide signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_32s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed complex short signal divide signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_16sc_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer signal divided by 16-bit signed short signal, scale, then clamp to 16-bit signed short saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_32s16s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit floating point signal divide signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_32f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point signal divide signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_64f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit complex floating point signal divide signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_32fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit complex floating point signal divide signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_64fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char in place signal divide signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_8u_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short in place signal divide signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_16u_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short in place signal divide signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_16s_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit complex signed short in place signal divide signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_16sc_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer in place signal divide signal, with scaling, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_32s_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit floating point in place signal divide signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_32f_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point in place signal divide signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_64f_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit complex floating point in place signal divide signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_32fc_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit complex floating point in place signal divide signal, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_64fc_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char signal divide signal, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nRndMode">various rounding modes.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_Round_8u_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            NppRoundMode nRndMode,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short signal divide signal, scale, round, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nRndMode">various rounding modes.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_Round_16u_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            NppRoundMode nRndMode,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short signal divide signal, scale, round, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer, signal1 divisor elements to be divided into signal2 dividend elements.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nRndMode">various rounding modes.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_Round_16s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength,
            NppRoundMode nRndMode,
            int nScaleFactor);

        /// <summary>
        /// 8-bit unsigned char in place signal divide signal, with scaling, rounding then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nRndMode">various rounding modes.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_Round_8u_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            NppRoundMode nRndMode,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short in place signal divide signal, with scaling, rounding then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nRndMode">various rounding modes.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_Round_16u_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            NppRoundMode nRndMode,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short in place signal divide signal, with scaling, rounding then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal1 divisor elements to be divided into signal2 dividend elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nRndMode">various rounding modes.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDiv_Round_16s_ISfs(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength,
            NppRoundMode nRndMode,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short signal absolute value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAbs_16s(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit signed integer signal absolute value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAbs_32s(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point signal absolute value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAbs_32f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point signal absolute value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAbs_64f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit signed short signal absolute value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAbs_16s_I(
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit signed integer signal absolute value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAbs_32s_I(
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point signal absolute value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAbs_32f_I(
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point signal absolute value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAbs_64f_I(
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point signal squared.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqr_32f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point signal squared.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqr_64f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit complex floating point signal squared.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqr_32fc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit complex floating point signal squared.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqr_64fc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point signal squared.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqr_32f_I(
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point signal squared.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqr_64f_I(
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit complex floating point signal squared.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqr_32fc_I(
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit complex floating point signal squared.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqr_64fc_I(
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char signal squared, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqr_8u_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short signal squared, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqr_16u_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short signal squared, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqr_16s_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit complex signed short signal squared, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqr_16sc_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 8-bit unsigned char signal squared, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqr_8u_ISfs(
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short signal squared, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqr_16u_ISfs(
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short signal squared, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqr_16s_ISfs(
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit complex signed short signal squared, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqr_16sc_ISfs(
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit floating point signal square root.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqrt_32f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point signal square root.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqrt_64f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit complex floating point signal square root.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqrt_32fc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit complex floating point signal square root.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqrt_64fc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point signal square root.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqrt_32f_I(
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point signal square root.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqrt_64f_I(
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit complex floating point signal square root.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqrt_32fc_I(
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit complex floating point signal square root.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqrt_64fc_I(
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char signal square root, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqrt_8u_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short signal square root, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqrt_16u_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short signal square root, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqrt_16s_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit complex signed short signal square root, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqrt_16sc_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 64-bit signed integer signal square root, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqrt_64s_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer signal square root, scale, then clamp to 16-bit signed integer saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqrt_32s16s_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 64-bit signed integer signal square root, scale, then clamp to 16-bit signed integer saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqrt_64s16s_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 8-bit unsigned char signal square root, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqrt_8u_ISfs(
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit unsigned short signal square root, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqrt_16u_ISfs(
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short signal square root, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqrt_16s_ISfs(
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit complex signed short signal square root, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqrt_16sc_ISfs(
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 64-bit signed integer signal square root, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSqrt_64s_ISfs(
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit floating point signal cube root.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsCubrt_32f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit signed integer signal cube root, scale, then clamp to 16-bit signed integer saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsCubrt_32s16s_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit floating point signal exponent.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsExp_32f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point signal exponent.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsExp_64f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point signal exponent with 64-bit floating point result.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsExp_32f64f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point signal exponent.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsExp_32f_I(
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point signal exponent.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsExp_64f_I(
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 16-bit signed short signal exponent, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsExp_16s_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer signal exponent, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsExp_32s_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 64-bit signed integer signal exponent, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsExp_64s_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short signal exponent, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsExp_16s_ISfs(
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer signal exponent, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsExp_32s_ISfs(
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 64-bit signed integer signal exponent, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsExp_64s_ISfs(
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit floating point signal natural logarithm.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsLn_32f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point signal natural logarithm.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsLn_64f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point signal natural logarithm with 32-bit floating point result.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsLn_64f32f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point signal natural logarithm.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsLn_32f_I(
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point signal natural logarithm.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsLn_64f_I(
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 16-bit signed short signal natural logarithm, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsLn_16s_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer signal natural logarithm, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsLn_32s_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer signal natural logarithm, scale, then clamp to 16-bit signed short saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsLn_32s16s_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short signal natural logarithm, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsLn_16s_ISfs(
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer signal natural logarithm, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsLn_32s_ISfs(
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer signal 10 times base 10 logarithm, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus npps10Log10_32s_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// 32-bit signed integer signal 10 times base 10 logarithm, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus npps10Log10_32s_ISfs(
            IntPtr pSrcDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// Device scratch buffer size (in bytes) for 32f SumLn. This primitive provides the correct buffer size for nppsSumLn_32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSumLnGetBufferSize_32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit floating point signal sum natural logarithm.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSumLn_32f(
            IntPtr pSrc,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device scratch buffer size (in bytes) for 64f SumLn. This primitive provides the correct buffer size for nppsSumLn_64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSumLnGetBufferSize_64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 64-bit floating point signal sum natural logarithm.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSumLn_64f(
            IntPtr pSrc,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device scratch buffer size (in bytes) for 32f64f SumLn. This primitive provides the correct buffer size for nppsSumLn_32f64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSumLnGetBufferSize_32f64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit flaoting point input, 64-bit floating point output signal sum natural logarithm.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSumLn_32f64f(
            IntPtr pSrc,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device scratch buffer size (in bytes) for 16s32f SumLn. This primitive provides the correct buffer size for nppsSumLn_16s32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSumLnGetBufferSize_16s32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer input, 32-bit floating point output signal sum natural logarithm.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSumLn_16s32f(
            IntPtr pSrc,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit floating point signal inverse tangent.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsArctan_32f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point signal inverse tangent.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsArctan_64f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point signal inverse tangent.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsArctan_32f_I(
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point signal inverse tangent.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsArctan_64f_I(
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point signal normalize.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="vSub">value subtracted from each signal element before division</param>
        /// <param name="vDiv">divisor of post-subtracted signal element dividend</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormalize_32f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp32f vSub,
            Npp32f vDiv);

        /// <summary>
        /// 32-bit complex floating point signal normalize.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="vSub">value subtracted from each signal element before division</param>
        /// <param name="vDiv">divisor of post-subtracted signal element dividend</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormalize_32fc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp32fc vSub,
            Npp32f vDiv);

        /// <summary>
        /// 64-bit floating point signal normalize.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="vSub">value subtracted from each signal element before division</param>
        /// <param name="vDiv">divisor of post-subtracted signal element dividend</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormalize_64f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp64f vSub,
            Npp64f vDiv);

        /// <summary>
        /// 64-bit complex floating point signal normalize.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="vSub">value subtracted from each signal element before division</param>
        /// <param name="vDiv">divisor of post-subtracted signal element dividend</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormalize_64fc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp64fc vSub,
            Npp64f vDiv);

        /// <summary>
        /// 16-bit signed short signal normalize, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="vSub">value subtracted from each signal element before division</param>
        /// <param name="vDiv">divisor of post-subtracted signal element dividend</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormalize_16s_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp16s vSub,
            int vDiv,
            int nScaleFactor);

        /// <summary>
        /// 16-bit complex signed short signal normalize, scale, then clamp to saturated value.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="vSub">value subtracted from each signal element before division</param>
        /// <param name="vDiv">divisor of post-subtracted signal element dividend</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormalize_16sc_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp16sc vSub,
            int vDiv,
            int nScaleFactor);

        /// <summary>
        /// 32-bit floating point signal Cauchy error calculation.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nParam">constant used in Cauchy formula</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsCauchy_32f_I(
            IntPtr pSrcDst,
            int nLength,
            Npp32f nParam);

        /// <summary>
        /// 32-bit floating point signal Cauchy first derivative.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nParam">constant used in Cauchy formula</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsCauchyD_32f_I(
            IntPtr pSrcDst,
            int nLength,
            Npp32f nParam);

        /// <summary>
        /// 32-bit floating point signal Cauchy first and second derivatives.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="pD2FVal">source_signal_pointer. This signal contains the second derivative of the source signal.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nParam">constant used in Cauchy formula</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsCauchyDD2_32f_I(
            IntPtr pSrcDst,
            IntPtr pD2FVal,
            int nLength,
            Npp32f nParam);

        /// <summary>
        /// 8-bit unsigned char signal and with constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be anded with each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAndC_8u(
            IntPtr pSrc,
            Npp8u nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned short signal and with constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be anded with each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAndC_16u(
            IntPtr pSrc,
            Npp16u nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit unsigned integer signal and with constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be anded with each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAndC_32u(
            IntPtr pSrc,
            Npp32u nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char in place signal and with constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be anded with each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAndC_8u_I(
            Npp8u nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned short in place signal and with constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be anded with each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAndC_16u_I(
            Npp16u nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit unsigned signed integer in place signal and with constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be anded with each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAndC_32u_I(
            Npp32u nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char signal and with signal.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be anded with signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAnd_8u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned short signal and with signal.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be anded with signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAnd_16u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit unsigned integer signal and with signal.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be anded with signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAnd_32u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char in place signal and with signal.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be anded with signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAnd_8u_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned short in place signal and with signal.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be anded with signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAnd_16u_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit unsigned integer in place signal and with signal.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be anded with signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAnd_32u_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char signal or with constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be ored with each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsOrC_8u(
            IntPtr pSrc,
            Npp8u nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned short signal or with constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be ored with each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsOrC_16u(
            IntPtr pSrc,
            Npp16u nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit unsigned integer signal or with constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be ored with each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsOrC_32u(
            IntPtr pSrc,
            Npp32u nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char in place signal or with constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be ored with each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsOrC_8u_I(
            Npp8u nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned short in place signal or with constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be ored with each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsOrC_16u_I(
            Npp16u nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit unsigned signed integer in place signal or with constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be ored with each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsOrC_32u_I(
            Npp32u nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char signal or with signal.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be ored with signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsOr_8u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned short signal or with signal.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be ored with signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsOr_16u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit unsigned integer signal or with signal.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be ored with signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsOr_32u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char in place signal or with signal.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be ored with signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsOr_8u_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned short in place signal or with signal.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be ored with signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsOr_16u_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit unsigned integer in place signal or with signal.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be ored with signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsOr_32u_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char signal exclusive or with constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be exclusive ored with each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsXorC_8u(
            IntPtr pSrc,
            Npp8u nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned short signal exclusive or with constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be exclusive ored with each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsXorC_16u(
            IntPtr pSrc,
            Npp16u nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit unsigned integer signal exclusive or with constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be exclusive ored with each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsXorC_32u(
            IntPtr pSrc,
            Npp32u nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char in place signal exclusive or with constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be exclusive ored with each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsXorC_8u_I(
            Npp8u nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned short in place signal exclusive or with constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be exclusive ored with each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsXorC_16u_I(
            Npp16u nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit unsigned signed integer in place signal exclusive or with constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be exclusive ored with each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsXorC_32u_I(
            Npp32u nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char signal exclusive or with signal.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be exclusive ored with signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsXor_8u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned short signal exclusive or with signal.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be exclusive ored with signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsXor_16u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit unsigned integer signal exclusive or with signal.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer. signal2 elements to be exclusive ored with signal1 elements</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsXor_32u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char in place signal exclusive or with signal.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be exclusive ored with signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsXor_8u_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned short in place signal exclusive or with signal.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be exclusive ored with signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsXor_16u_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit unsigned integer in place signal exclusive or with signal.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer. signal2 elements to be exclusive ored with signal1 elements</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsXor_32u_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char not signal.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNot_8u(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned short not signal.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNot_16u(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit unsigned integer not signal.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNot_32u(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char in place not signal.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNot_8u_I(
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned short in place not signal.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNot_16u_I(
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit unsigned signed integer in place not signal.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNot_32u_I(
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char signal left shift with constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be used to left shift each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsLShiftC_8u(
            IntPtr pSrc,
            int nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned short signal left shift with constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be used to left shift each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsLShiftC_16u(
            IntPtr pSrc,
            int nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit signed short signal left shift with constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be used to left shift each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsLShiftC_16s(
            IntPtr pSrc,
            int nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit unsigned integer signal left shift with constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be used to left shift each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsLShiftC_32u(
            IntPtr pSrc,
            int nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit signed integer signal left shift with constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be used to left shift each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsLShiftC_32s(
            IntPtr pSrc,
            int nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char in place signal left shift with constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be used to left shift each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsLShiftC_8u_I(
            int nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned short in place signal left shift with constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be used to left shift each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsLShiftC_16u_I(
            int nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 16-bit signed short in place signal left shift with constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be used to left shift each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsLShiftC_16s_I(
            int nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit unsigned signed integer in place signal left shift with constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be used to left shift each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsLShiftC_32u_I(
            int nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit signed signed integer in place signal left shift with constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be used to left shift each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsLShiftC_32s_I(
            int nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char signal right shift with constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be used to right shift each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsRShiftC_8u(
            IntPtr pSrc,
            int nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned short signal right shift with constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be used to right shift each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsRShiftC_16u(
            IntPtr pSrc,
            int nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 16-bit signed short signal right shift with constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be used to right shift each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsRShiftC_16s(
            IntPtr pSrc,
            int nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit unsigned integer signal right shift with constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be used to right shift each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsRShiftC_32u(
            IntPtr pSrc,
            int nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 32-bit signed integer signal right shift with constant.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nValue">Constant value to be used to right shift each vector element</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsRShiftC_32s(
            IntPtr pSrc,
            int nValue,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// 8-bit unsigned char in place signal right shift with constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be used to right shift each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsRShiftC_8u_I(
            int nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned short in place signal right shift with constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be used to right shift each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsRShiftC_16u_I(
            int nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 16-bit signed short in place signal right shift with constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be used to right shift each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsRShiftC_16s_I(
            int nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit unsigned signed integer in place signal right shift with constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be used to right shift each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsRShiftC_32u_I(
            int nValue,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit signed signed integer in place signal right shift with constant.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nValue">Constant value to be used to right shift each vector element</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsRShiftC_32s_I(
            int nValue,
            IntPtr pSrcDst,
            int nLength);

    }
}