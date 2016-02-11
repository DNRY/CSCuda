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
        public static extern NppStatus nppsConvert_8s16s(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_8s32f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_8u32f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_16s8s_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            Npp32u nLength,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_16s32s(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_16s32f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_16u32f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_32s16s(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_32s32f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_32s64f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_32f64f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_64s64f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_64f32f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_16s32f_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_16s64f_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_32s16s_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_32s32f_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_32s64f_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            int nScaleFactor);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_32f8s_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_32f8u_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_32f16s_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_32f16u_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_32f32s_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_64s32s_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_64f16s_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_64f32s_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// </summary>
        /// <returns></returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsConvert_64f64s_Sfs(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            NppRoundMode eRoundMode,
            int nScaleFactor);

        /// <summary>
        /// 16-bit signed short signal threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_16s(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp16s nLevel,
            NppCmpOp nRelOp);

        /// <summary>
        /// 16-bit in place signed short signal threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_16s_I(
            IntPtr pSrcDst,
            int nLength,
            Npp16s nLevel,
            NppCmpOp nRelOp);

        /// <summary>
        /// 16-bit signed short complex number signal threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_16sc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp16s nLevel,
            NppCmpOp nRelOp);

        /// <summary>
        /// 16-bit in place signed short complex number signal threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_16sc_I(
            IntPtr pSrcDst,
            int nLength,
            Npp16s nLevel,
            NppCmpOp nRelOp);

        /// <summary>
        /// 32-bit floating point signal threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_32f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp32f nLevel,
            NppCmpOp nRelOp);

        /// <summary>
        /// 32-bit in place floating point signal threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_32f_I(
            IntPtr pSrcDst,
            int nLength,
            Npp32f nLevel,
            NppCmpOp nRelOp);

        /// <summary>
        /// 32-bit floating point complex number signal threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_32fc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp32f nLevel,
            NppCmpOp nRelOp);

        /// <summary>
        /// 32-bit in place floating point complex number signal threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_32fc_I(
            IntPtr pSrcDst,
            int nLength,
            Npp32f nLevel,
            NppCmpOp nRelOp);

        /// <summary>
        /// 64-bit floating point signal threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_64f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp64f nLevel,
            NppCmpOp nRelOp);

        /// <summary>
        /// 64-bit in place floating point signal threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_64f_I(
            IntPtr pSrcDst,
            int nLength,
            Npp64f nLevel,
            NppCmpOp nRelOp);

        /// <summary>
        /// 64-bit floating point complex number signal threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_64fc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp64f nLevel,
            NppCmpOp nRelOp);

        /// <summary>
        /// 64-bit in place floating point complex number signal threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <param name="nRelOp">NppCmpOp type of thresholding operation (NPP_CMP_LESS or NPP_CMP_GREATER only).</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_64fc_I(
            IntPtr pSrcDst,
            int nLength,
            Npp64f nLevel,
            NppCmpOp nRelOp);

        /// <summary>
        /// 16-bit signed short signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LT_16s(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp16s nLevel);

        /// <summary>
        /// 16-bit in place signed short signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LT_16s_I(
            IntPtr pSrcDst,
            int nLength,
            Npp16s nLevel);

        /// <summary>
        /// 16-bit signed short complex number signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LT_16sc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp16s nLevel);

        /// <summary>
        /// 16-bit in place signed short complex number signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LT_16sc_I(
            IntPtr pSrcDst,
            int nLength,
            Npp16s nLevel);

        /// <summary>
        /// 32-bit floating point signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LT_32f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp32f nLevel);

        /// <summary>
        /// 32-bit in place floating point signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LT_32f_I(
            IntPtr pSrcDst,
            int nLength,
            Npp32f nLevel);

        /// <summary>
        /// 32-bit floating point complex number signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LT_32fc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp32f nLevel);

        /// <summary>
        /// 32-bit in place floating point complex number signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LT_32fc_I(
            IntPtr pSrcDst,
            int nLength,
            Npp32f nLevel);

        /// <summary>
        /// 64-bit floating point signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LT_64f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp64f nLevel);

        /// <summary>
        /// 64-bit in place floating point signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LT_64f_I(
            IntPtr pSrcDst,
            int nLength,
            Npp64f nLevel);

        /// <summary>
        /// 64-bit floating point complex number signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LT_64fc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp64f nLevel);

        /// <summary>
        /// 64-bit in place floating point complex number signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LT_64fc_I(
            IntPtr pSrcDst,
            int nLength,
            Npp64f nLevel);

        /// <summary>
        /// 16-bit signed short signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GT_16s(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp16s nLevel);

        /// <summary>
        /// 16-bit in place signed short signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GT_16s_I(
            IntPtr pSrcDst,
            int nLength,
            Npp16s nLevel);

        /// <summary>
        /// 16-bit signed short complex number signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GT_16sc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp16s nLevel);

        /// <summary>
        /// 16-bit in place signed short complex number signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GT_16sc_I(
            IntPtr pSrcDst,
            int nLength,
            Npp16s nLevel);

        /// <summary>
        /// 32-bit floating point signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GT_32f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp32f nLevel);

        /// <summary>
        /// 32-bit in place floating point signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GT_32f_I(
            IntPtr pSrcDst,
            int nLength,
            Npp32f nLevel);

        /// <summary>
        /// 32-bit floating point complex number signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GT_32fc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp32f nLevel);

        /// <summary>
        /// 32-bit in place floating point complex number signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GT_32fc_I(
            IntPtr pSrcDst,
            int nLength,
            Npp32f nLevel);

        /// <summary>
        /// 64-bit floating point signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GT_64f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp64f nLevel);

        /// <summary>
        /// 64-bit in place floating point signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GT_64f_I(
            IntPtr pSrcDst,
            int nLength,
            Npp64f nLevel);

        /// <summary>
        /// 64-bit floating point complex number signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GT_64fc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp64f nLevel);

        /// <summary>
        /// 64-bit in place floating point complex number signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GT_64fc_I(
            IntPtr pSrcDst,
            int nLength,
            Npp64f nLevel);

        /// <summary>
        /// 16-bit signed short signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LTVal_16s(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp16s nLevel,
            Npp16s nValue);

        /// <summary>
        /// 16-bit in place signed short signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LTVal_16s_I(
            IntPtr pSrcDst,
            int nLength,
            Npp16s nLevel,
            Npp16s nValue);

        /// <summary>
        /// 16-bit signed short complex number signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LTVal_16sc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp16s nLevel,
            Npp16sc nValue);

        /// <summary>
        /// 16-bit in place signed short complex number signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LTVal_16sc_I(
            IntPtr pSrcDst,
            int nLength,
            Npp16s nLevel,
            Npp16sc nValue);

        /// <summary>
        /// 32-bit floating point signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LTVal_32f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp32f nLevel,
            Npp32f nValue);

        /// <summary>
        /// 32-bit in place floating point signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LTVal_32f_I(
            IntPtr pSrcDst,
            int nLength,
            Npp32f nLevel,
            Npp32f nValue);

        /// <summary>
        /// 32-bit floating point complex number signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LTVal_32fc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp32f nLevel,
            Npp32fc nValue);

        /// <summary>
        /// 32-bit in place floating point complex number signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LTVal_32fc_I(
            IntPtr pSrcDst,
            int nLength,
            Npp32f nLevel,
            Npp32fc nValue);

        /// <summary>
        /// 64-bit floating point signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LTVal_64f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp64f nLevel,
            Npp64f nValue);

        /// <summary>
        /// 64-bit in place floating point signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LTVal_64f_I(
            IntPtr pSrcDst,
            int nLength,
            Npp64f nLevel,
            Npp64f nValue);

        /// <summary>
        /// 64-bit floating point complex number signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LTVal_64fc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp64f nLevel,
            Npp64fc nValue);

        /// <summary>
        /// 64-bit in place floating point complex number signal NPP_CMP_LESS threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_LTVal_64fc_I(
            IntPtr pSrcDst,
            int nLength,
            Npp64f nLevel,
            Npp64fc nValue);

        /// <summary>
        /// 16-bit signed short signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GTVal_16s(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp16s nLevel,
            Npp16s nValue);

        /// <summary>
        /// 16-bit in place signed short signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GTVal_16s_I(
            IntPtr pSrcDst,
            int nLength,
            Npp16s nLevel,
            Npp16s nValue);

        /// <summary>
        /// 16-bit signed short complex number signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GTVal_16sc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp16s nLevel,
            Npp16sc nValue);

        /// <summary>
        /// 16-bit in place signed short complex number signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GTVal_16sc_I(
            IntPtr pSrcDst,
            int nLength,
            Npp16s nLevel,
            Npp16sc nValue);

        /// <summary>
        /// 32-bit floating point signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GTVal_32f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp32f nLevel,
            Npp32f nValue);

        /// <summary>
        /// 32-bit in place floating point signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GTVal_32f_I(
            IntPtr pSrcDst,
            int nLength,
            Npp32f nLevel,
            Npp32f nValue);

        /// <summary>
        /// 32-bit floating point complex number signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GTVal_32fc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp32f nLevel,
            Npp32fc nValue);

        /// <summary>
        /// 32-bit in place floating point complex number signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GTVal_32fc_I(
            IntPtr pSrcDst,
            int nLength,
            Npp32f nLevel,
            Npp32fc nValue);

        /// <summary>
        /// 64-bit floating point signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GTVal_64f(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp64f nLevel,
            Npp64f nValue);

        /// <summary>
        /// 64-bit in place floating point signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GTVal_64f_I(
            IntPtr pSrcDst,
            int nLength,
            Npp64f nLevel,
            Npp64f nValue);

        /// <summary>
        /// 64-bit floating point complex number signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pDst">destination_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GTVal_64fc(
            IntPtr pSrc,
            IntPtr pDst,
            int nLength,
            Npp64f nLevel,
            Npp64fc nValue);

        /// <summary>
        /// 64-bit in place floating point complex number signal NPP_CMP_GREATER threshold with constant level.
        /// </summary>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="nLevel">Constant threshold value (real part only and must be greater than 0) to be used to limit each signal sample</param>
        /// <param name="nValue">Constant value to replace source value when threshold test is true.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsThreshold_GTVal_64fc_I(
            IntPtr pSrcDst,
            int nLength,
            Npp64f nLevel,
            Npp64fc nValue);

    }
}