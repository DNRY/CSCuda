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
        /// 8-bit in place min value for each pair of elements.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinEvery_8u_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned short integer in place min value for each pair of elements.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinEvery_16u_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 16-bit signed short integer in place min value for each pair of elements.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinEvery_16s_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit signed integer in place min value for each pair of elements.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinEvery_32s_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point in place min value for each pair of elements.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinEvery_32f_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 64-bit floating point in place min value for each pair of elements.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinEvery_64f_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 8-bit in place max value for each pair of elements.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxEvery_8u_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 16-bit unsigned short integer in place max value for each pair of elements.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxEvery_16u_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 16-bit signed short integer in place max value for each pair of elements.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxEvery_16s_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit signed integer in place max value for each pair of elements.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxEvery_32s_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// 32-bit floating point in place max value for each pair of elements.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="pSrcDst">in_place_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxEvery_32f_I(
            IntPtr pSrc,
            IntPtr pSrcDst,
            int nLength);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsSum_32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSumGetBufferSize_32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsSum_32fc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSumGetBufferSize_32fc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsSum_64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSumGetBufferSize_64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsSum_64fc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSumGetBufferSize_64fc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsSum_16s_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSumGetBufferSize_16s_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsSum_16sc_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSumGetBufferSize_16sc_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsSum_16sc32sc_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSumGetBufferSize_16sc32sc_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsSum_32s_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSumGetBufferSize_32s_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsSum_16s32s_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSumGetBufferSize_16s32s_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit float vector sum method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pSum">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSum_32f(
            IntPtr pSrc,
            int nLength,
            IntPtr pSum,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit float complex vector sum method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pSum">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSum_32fc(
            IntPtr pSrc,
            int nLength,
            IntPtr pSum,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit double vector sum method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pSum">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSum_64f(
            IntPtr pSrc,
            int nLength,
            IntPtr pSum,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit double complex vector sum method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pSum">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSum_64fc(
            IntPtr pSrc,
            int nLength,
            IntPtr pSum,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit short vector sum with integer scaling method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pSum">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSum_16s_Sfs(
            IntPtr pSrc,
            int nLength,
            IntPtr pSum,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit integer vector sum with integer scaling method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pSum">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSum_32s_Sfs(
            IntPtr pSrc,
            int nLength,
            IntPtr pSum,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit short complex vector sum with integer scaling method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pSum">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSum_16sc_Sfs(
            IntPtr pSrc,
            int nLength,
            IntPtr pSum,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit short complex vector sum (32bit int complex) with integer scaling method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pSum">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSum_16sc32sc_Sfs(
            IntPtr pSrc,
            int nLength,
            IntPtr pSum,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit integer vector sum (32bit) with integer scaling method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pSum">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsSum_16s32s_Sfs(
            IntPtr pSrc,
            int nLength,
            IntPtr pSum,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMax_16s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxGetBufferSize_16s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMax_32s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxGetBufferSize_32s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMax_32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxGetBufferSize_32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMax_64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxGetBufferSize_64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit integer vector max method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMax">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMax_16s(
            IntPtr pSrc,
            int nLength,
            IntPtr pMax,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit integer vector max method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMax">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMax_32s(
            IntPtr pSrc,
            int nLength,
            IntPtr pMax,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit float vector max method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMax">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMax_32f(
            IntPtr pSrc,
            int nLength,
            IntPtr pMax,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit float vector max method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMax">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMax_64f(
            IntPtr pSrc,
            int nLength,
            IntPtr pMax,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMaxIndx_16s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxIndxGetBufferSize_16s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMaxIndx_32s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxIndxGetBufferSize_32s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMaxIndx_32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxIndxGetBufferSize_32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMaxIndx_64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxIndxGetBufferSize_64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit integer vector max index method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMax">Pointer to the output result.</param>
        /// <param name="pIndx">Pointer to the index value of the first maximum element.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxIndx_16s(
            IntPtr pSrc,
            int nLength,
            IntPtr pMax,
            IntPtr pIndx,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit integer vector max index method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMax">Pointer to the output result.</param>
        /// <param name="pIndx">Pointer to the index value of the first maximum element.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxIndx_32s(
            IntPtr pSrc,
            int nLength,
            IntPtr pMax,
            IntPtr pIndx,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit float vector max index method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMax">Pointer to the output result.</param>
        /// <param name="pIndx">Pointer to the index value of the first maximum element.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxIndx_32f(
            IntPtr pSrc,
            int nLength,
            IntPtr pMax,
            IntPtr pIndx,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit float vector max index method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMax">Pointer to the output result.</param>
        /// <param name="pIndx">Pointer to the index value of the first maximum element.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxIndx_64f(
            IntPtr pSrc,
            int nLength,
            IntPtr pMax,
            IntPtr pIndx,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMaxAbs_16s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxAbsGetBufferSize_16s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMaxAbs_32s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxAbsGetBufferSize_32s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit integer vector max absolute method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMaxAbs">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxAbs_16s(
            IntPtr pSrc,
            int nLength,
            IntPtr pMaxAbs,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit integer vector max absolute method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMaxAbs">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxAbs_32s(
            IntPtr pSrc,
            int nLength,
            IntPtr pMaxAbs,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMaxAbsIndx_16s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxAbsIndxGetBufferSize_16s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMaxAbsIndx_32s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxAbsIndxGetBufferSize_32s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit integer vector max absolute index method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMaxAbs">Pointer to the output result.</param>
        /// <param name="pIndx">Pointer to the index value of the first maximum element.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxAbsIndx_16s(
            IntPtr pSrc,
            int nLength,
            IntPtr pMaxAbs,
            IntPtr pIndx,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit integer vector max absolute index method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMaxAbs">Pointer to the output result.</param>
        /// <param name="pIndx">Pointer to the index value of the first maximum element.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaxAbsIndx_32s(
            IntPtr pSrc,
            int nLength,
            IntPtr pMaxAbs,
            IntPtr pIndx,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMin_16s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinGetBufferSize_16s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMin_32s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinGetBufferSize_32s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMin_32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinGetBufferSize_32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMin_64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinGetBufferSize_64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit integer vector min method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMin">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMin_16s(
            IntPtr pSrc,
            int nLength,
            IntPtr pMin,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit integer vector min method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMin">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMin_32s(
            IntPtr pSrc,
            int nLength,
            IntPtr pMin,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit integer vector min method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMin">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMin_32f(
            IntPtr pSrc,
            int nLength,
            IntPtr pMin,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit integer vector min method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMin">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMin_64f(
            IntPtr pSrc,
            int nLength,
            IntPtr pMin,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMinIndx_16s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinIndxGetBufferSize_16s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMinIndx_32s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinIndxGetBufferSize_32s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMinIndx_32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinIndxGetBufferSize_32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMinIndx_64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinIndxGetBufferSize_64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit integer vector min index method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMin">Pointer to the output result.</param>
        /// <param name="pIndx">Pointer to the index value of the first minimum element.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinIndx_16s(
            IntPtr pSrc,
            int nLength,
            IntPtr pMin,
            IntPtr pIndx,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit integer vector min index method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMin">Pointer to the output result.</param>
        /// <param name="pIndx">Pointer to the index value of the first minimum element.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinIndx_32s(
            IntPtr pSrc,
            int nLength,
            IntPtr pMin,
            IntPtr pIndx,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit float vector min index method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMin">Pointer to the output result.</param>
        /// <param name="pIndx">Pointer to the index value of the first minimum element.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinIndx_32f(
            IntPtr pSrc,
            int nLength,
            IntPtr pMin,
            IntPtr pIndx,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit float vector min index method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMin">Pointer to the output result.</param>
        /// <param name="pIndx">Pointer to the index value of the first minimum element.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinIndx_64f(
            IntPtr pSrc,
            int nLength,
            IntPtr pMin,
            IntPtr pIndx,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMinAbs_16s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinAbsGetBufferSize_16s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMinAbs_32s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinAbsGetBufferSize_32s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit integer vector min absolute method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMinAbs">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinAbs_16s(
            IntPtr pSrc,
            int nLength,
            IntPtr pMinAbs,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit integer vector min absolute method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMinAbs">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinAbs_32s(
            IntPtr pSrc,
            int nLength,
            IntPtr pMinAbs,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMinAbsIndx_16s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinAbsIndxGetBufferSize_16s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMinAbsIndx_32s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinAbsIndxGetBufferSize_32s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit integer vector min absolute index method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMinAbs">Pointer to the output result.</param>
        /// <param name="pIndx">Pointer to the index value of the first minimum element.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinAbsIndx_16s(
            IntPtr pSrc,
            int nLength,
            IntPtr pMinAbs,
            IntPtr pIndx,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit integer vector min absolute index method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMinAbs">Pointer to the output result.</param>
        /// <param name="pIndx">Pointer to the index value of the first minimum element.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinAbsIndx_32s(
            IntPtr pSrc,
            int nLength,
            IntPtr pMinAbs,
            IntPtr pIndx,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMean_32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMeanGetBufferSize_32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMean_32fc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMeanGetBufferSize_32fc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMean_64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMeanGetBufferSize_64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMean_64fc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMeanGetBufferSize_64fc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMean_16s_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMeanGetBufferSize_16s_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMean_32s_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMeanGetBufferSize_32s_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMean_16sc_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMeanGetBufferSize_16sc_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit float vector mean method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMean">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMean_32f(
            IntPtr pSrc,
            int nLength,
            IntPtr pMean,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit float complex vector mean method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMean">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMean_32fc(
            IntPtr pSrc,
            int nLength,
            IntPtr pMean,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit double vector mean method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMean">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMean_64f(
            IntPtr pSrc,
            int nLength,
            IntPtr pMean,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit double complex vector mean method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMean">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMean_64fc(
            IntPtr pSrc,
            int nLength,
            IntPtr pMean,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit short vector mean with integer scaling method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMean">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMean_16s_Sfs(
            IntPtr pSrc,
            int nLength,
            IntPtr pMean,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit integer vector mean with integer scaling method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMean">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMean_32s_Sfs(
            IntPtr pSrc,
            int nLength,
            IntPtr pMean,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit short complex vector mean with integer scaling method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMean">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMean_16sc_Sfs(
            IntPtr pSrc,
            int nLength,
            IntPtr pMean,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsStdDev_32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsStdDevGetBufferSize_32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsStdDev_64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsStdDevGetBufferSize_64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsStdDev_16s32s_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsStdDevGetBufferSize_16s32s_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsStdDev_16s_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsStdDevGetBufferSize_16s_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit float vector standard deviation method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pStdDev">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsStdDev_32f(
            IntPtr pSrc,
            int nLength,
            IntPtr pStdDev,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit float vector standard deviation method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pStdDev">Pointer to the output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsStdDev_64f(
            IntPtr pSrc,
            int nLength,
            IntPtr pStdDev,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit float vector standard deviation method (return value is 32-bit)
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pStdDev">Pointer to the output result.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsStdDev_16s32s_Sfs(
            IntPtr pSrc,
            int nLength,
            IntPtr pStdDev,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit float vector standard deviation method (return value is also 16-bit)
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pStdDev">Pointer to the output result.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsStdDev_16s_Sfs(
            IntPtr pSrc,
            int nLength,
            IntPtr pStdDev,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMeanStdDev_32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMeanStdDevGetBufferSize_32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMeanStdDev_64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMeanStdDevGetBufferSize_64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMeanStdDev_16s32s_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMeanStdDevGetBufferSize_16s32s_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device scratch buffer size (in bytes) for nppsMeanStdDev_16s_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMeanStdDevGetBufferSize_16s_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit float vector mean and standard deviation method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMean">Pointer to the output mean value.</param>
        /// <param name="pStdDev">Pointer to the output standard deviation value.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMeanStdDev_32f(
            IntPtr pSrc,
            int nLength,
            IntPtr pMean,
            IntPtr pStdDev,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit float vector mean and standard deviation method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMean">Pointer to the output mean value.</param>
        /// <param name="pStdDev">Pointer to the output standard deviation value.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMeanStdDev_64f(
            IntPtr pSrc,
            int nLength,
            IntPtr pMean,
            IntPtr pStdDev,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit float vector mean and standard deviation method (return values are 32-bit)
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMean">Pointer to the output mean value.</param>
        /// <param name="pStdDev">Pointer to the output standard deviation value.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMeanStdDev_16s32s_Sfs(
            IntPtr pSrc,
            int nLength,
            IntPtr pMean,
            IntPtr pStdDev,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit float vector mean and standard deviation method (return values are also 16-bit)
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMean">Pointer to the output mean value.</param>
        /// <param name="pStdDev">Pointer to the output standard deviation value.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMeanStdDev_16s_Sfs(
            IntPtr pSrc,
            int nLength,
            IntPtr pMean,
            IntPtr pStdDev,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMinMax_8u.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMaxGetBufferSize_8u(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMinMax_16s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMaxGetBufferSize_16s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMinMax_16u.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMaxGetBufferSize_16u(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMinMax_32s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMaxGetBufferSize_32s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMinMax_32u.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMaxGetBufferSize_32u(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMinMax_32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMaxGetBufferSize_32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMinMax_64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMaxGetBufferSize_64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 8-bit char vector min and max method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMin">Pointer to the min output result.</param>
        /// <param name="pMax">Pointer to the max output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMax_8u(
            IntPtr pSrc,
            int nLength,
            IntPtr pMin,
            IntPtr pMax,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit signed short vector min and max method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMin">Pointer to the min output result.</param>
        /// <param name="pMax">Pointer to the max output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMax_16s(
            IntPtr pSrc,
            int nLength,
            IntPtr pMin,
            IntPtr pMax,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit unsigned short vector min and max method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMin">Pointer to the min output result.</param>
        /// <param name="pMax">Pointer to the max output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMax_16u(
            IntPtr pSrc,
            int nLength,
            IntPtr pMin,
            IntPtr pMax,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit unsigned int vector min and max method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMin">Pointer to the min output result.</param>
        /// <param name="pMax">Pointer to the max output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMax_32u(
            IntPtr pSrc,
            int nLength,
            IntPtr pMin,
            IntPtr pMax,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit signed int vector min and max method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMin">Pointer to the min output result.</param>
        /// <param name="pMax">Pointer to the max output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMax_32s(
            IntPtr pSrc,
            int nLength,
            IntPtr pMin,
            IntPtr pMax,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit float vector min and max method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMin">Pointer to the min output result.</param>
        /// <param name="pMax">Pointer to the max output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMax_32f(
            IntPtr pSrc,
            int nLength,
            IntPtr pMin,
            IntPtr pMax,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit double vector min and max method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMin">Pointer to the min output result.</param>
        /// <param name="pMax">Pointer to the max output result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMax_64f(
            IntPtr pSrc,
            int nLength,
            IntPtr pMin,
            IntPtr pMax,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMinMaxIndx_8u.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMaxIndxGetBufferSize_8u(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMinMaxIndx_16s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMaxIndxGetBufferSize_16s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMinMaxIndx_16u.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMaxIndxGetBufferSize_16u(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMinMaxIndx_32s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMaxIndxGetBufferSize_32s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMinMaxIndx_32u.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMaxIndxGetBufferSize_32u(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMinMaxIndx_32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMaxIndxGetBufferSize_32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMinMaxIndx_64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMaxIndxGetBufferSize_64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 8-bit char vector min and max with indices method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMin">Pointer to the min output result.</param>
        /// <param name="pMinIndx">Pointer to the index of the first min value.</param>
        /// <param name="pMax">Pointer to the max output result.</param>
        /// <param name="pMaxIndx">Pointer to the index of the first max value.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMaxIndx_8u(
            IntPtr pSrc,
            int nLength,
            IntPtr pMin,
            IntPtr pMinIndx,
            IntPtr pMax,
            IntPtr pMaxIndx,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit signed short vector min and max with indices method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMin">Pointer to the min output result.</param>
        /// <param name="pMinIndx">Pointer to the index of the first min value.</param>
        /// <param name="pMax">Pointer to the max output result.</param>
        /// <param name="pMaxIndx">Pointer to the index of the first max value.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMaxIndx_16s(
            IntPtr pSrc,
            int nLength,
            IntPtr pMin,
            IntPtr pMinIndx,
            IntPtr pMax,
            IntPtr pMaxIndx,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit unsigned short vector min and max with indices method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMin">Pointer to the min output result.</param>
        /// <param name="pMinIndx">Pointer to the index of the first min value.</param>
        /// <param name="pMax">Pointer to the max output result.</param>
        /// <param name="pMaxIndx">Pointer to the index of the first max value.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMaxIndx_16u(
            IntPtr pSrc,
            int nLength,
            IntPtr pMin,
            IntPtr pMinIndx,
            IntPtr pMax,
            IntPtr pMaxIndx,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit signed short vector min and max with indices method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMin">Pointer to the min output result.</param>
        /// <param name="pMinIndx">Pointer to the index of the first min value.</param>
        /// <param name="pMax">Pointer to the max output result.</param>
        /// <param name="pMaxIndx">Pointer to the index of the first max value.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMaxIndx_32s(
            IntPtr pSrc,
            int nLength,
            IntPtr pMin,
            IntPtr pMinIndx,
            IntPtr pMax,
            IntPtr pMaxIndx,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit unsigned short vector min and max with indices method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMin">Pointer to the min output result.</param>
        /// <param name="pMinIndx">Pointer to the index of the first min value.</param>
        /// <param name="pMax">Pointer to the max output result.</param>
        /// <param name="pMaxIndx">Pointer to the index of the first max value.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMaxIndx_32u(
            IntPtr pSrc,
            int nLength,
            IntPtr pMin,
            IntPtr pMinIndx,
            IntPtr pMax,
            IntPtr pMaxIndx,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit float vector min and max with indices method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMin">Pointer to the min output result.</param>
        /// <param name="pMinIndx">Pointer to the index of the first min value.</param>
        /// <param name="pMax">Pointer to the max output result.</param>
        /// <param name="pMaxIndx">Pointer to the index of the first max value.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMaxIndx_32f(
            IntPtr pSrc,
            int nLength,
            IntPtr pMin,
            IntPtr pMinIndx,
            IntPtr pMax,
            IntPtr pMaxIndx,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit float vector min and max with indices method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pMin">Pointer to the min output result.</param>
        /// <param name="pMinIndx">Pointer to the index of the first min value.</param>
        /// <param name="pMax">Pointer to the max output result.</param>
        /// <param name="pMaxIndx">Pointer to the index of the first max value.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMinMaxIndx_64f(
            IntPtr pSrc,
            int nLength,
            IntPtr pMin,
            IntPtr pMinIndx,
            IntPtr pMax,
            IntPtr pMaxIndx,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNorm_Inf_32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormInfGetBufferSize_32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit float vector C norm method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNorm_Inf_32f(
            IntPtr pSrc,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNorm_Inf_64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormInfGetBufferSize_64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 64-bit float vector C norm method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNorm_Inf_64f(
            IntPtr pSrc,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNorm_Inf_16s32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormInfGetBufferSize_16s32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer vector C norm method, return value is 32-bit float.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNorm_Inf_16s32f(
            IntPtr pSrc,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNorm_Inf_32fc32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormInfGetBufferSize_32fc32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit float complex vector C norm method, return value is 32-bit float.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNorm_Inf_32fc32f(
            IntPtr pSrc,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNorm_Inf_64fc64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormInfGetBufferSize_64fc64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 64-bit float complex vector C norm method, return value is 64-bit float.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNorm_Inf_64fc64f(
            IntPtr pSrc,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNorm_Inf_16s32s_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormInfGetBufferSize_16s32s_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer vector C norm method, return value is 32-bit signed integer.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNorm_Inf_16s32s_Sfs(
            IntPtr pSrc,
            int nLength,
            IntPtr pNorm,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNorm_L1_32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormL1GetBufferSize_32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit float vector L1 norm method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNorm_L1_32f(
            IntPtr pSrc,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNorm_L1_64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormL1GetBufferSize_64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 64-bit float vector L1 norm method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNorm_L1_64f(
            IntPtr pSrc,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNorm_L1_16s32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormL1GetBufferSize_16s32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer vector L1 norm method, return value is 32-bit float.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the L1 norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNorm_L1_16s32f(
            IntPtr pSrc,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNorm_L1_32fc64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormL1GetBufferSize_32fc64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit float complex vector L1 norm method, return value is 64-bit float.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNorm_L1_32fc64f(
            IntPtr pSrc,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNorm_L1_64fc64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormL1GetBufferSize_64fc64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 64-bit float complex vector L1 norm method, return value is 64-bit float.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNorm_L1_64fc64f(
            IntPtr pSrc,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNorm_L1_16s32s_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormL1GetBufferSize_16s32s_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer vector L1 norm method, return value is 32-bit signed integer.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNorm_L1_16s32s_Sfs(
            IntPtr pSrc,
            int nLength,
            IntPtr pNorm,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNorm_L1_16s64s_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormL1GetBufferSize_16s64s_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer vector L1 norm method, return value is 64-bit signed integer.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNorm_L1_16s64s_Sfs(
            IntPtr pSrc,
            int nLength,
            IntPtr pNorm,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNorm_L2_32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormL2GetBufferSize_32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit float vector L2 norm method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNorm_L2_32f(
            IntPtr pSrc,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNorm_L2_64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormL2GetBufferSize_64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 64-bit float vector L2 norm method
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNorm_L2_64f(
            IntPtr pSrc,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNorm_L2_16s32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormL2GetBufferSize_16s32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer vector L2 norm method, return value is 32-bit float.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNorm_L2_16s32f(
            IntPtr pSrc,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNorm_L2_32fc64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormL2GetBufferSize_32fc64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit float complex vector L2 norm method, return value is 64-bit float.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNorm_L2_32fc64f(
            IntPtr pSrc,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNorm_L2_64fc64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormL2GetBufferSize_64fc64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 64-bit float complex vector L2 norm method, return value is 64-bit float.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNorm_L2_64fc64f(
            IntPtr pSrc,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNorm_L2_16s32s_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormL2GetBufferSize_16s32s_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer vector L2 norm method, return value is 32-bit signed integer.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNorm_L2_16s32s_Sfs(
            IntPtr pSrc,
            int nLength,
            IntPtr pNorm,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNorm_L2Sqr_16s64s_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormL2SqrGetBufferSize_16s64s_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer vector L2 Square norm method, return value is 64-bit signed integer.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNorm_L2Sqr_16s64s_Sfs(
            IntPtr pSrc,
            int nLength,
            IntPtr pNorm,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNormDiff_Inf_32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiffInfGetBufferSize_32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit float C norm method on two vectors' difference
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiff_Inf_32f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNormDiff_Inf_64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiffInfGetBufferSize_64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 64-bit float C norm method on two vectors' difference
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiff_Inf_64f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNormDiff_Inf_16s32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiffInfGetBufferSize_16s32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer C norm method on two vectors' difference, return value is 32-bit float.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiff_Inf_16s32f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNormDiff_Inf_32fc32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiffInfGetBufferSize_32fc32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit float complex C norm method on two vectors' difference, return value is 32-bit float.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiff_Inf_32fc32f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNormDiff_Inf_64fc64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiffInfGetBufferSize_64fc64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 64-bit float complex C norm method on two vectors' difference, return value is 64-bit float.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiff_Inf_64fc64f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNormDiff_Inf_16s32s_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiffInfGetBufferSize_16s32s_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer C norm method on two vectors' difference, return value is 32-bit signed integer.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiff_Inf_16s32s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pNorm,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNormDiff_L1_32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiffL1GetBufferSize_32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit float L1 norm method on two vectors' difference
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiff_L1_32f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNormDiff_L1_64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiffL1GetBufferSize_64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 64-bit float L1 norm method on two vectors' difference
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiff_L1_64f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNormDiff_L1_16s32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiffL1GetBufferSize_16s32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer L1 norm method on two vectors' difference, return value is 32-bit float.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the L1 norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiff_L1_16s32f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNormDiff_L1_32fc64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiffL1GetBufferSize_32fc64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit float complex L1 norm method on two vectors' difference, return value is 64-bit float.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiff_L1_32fc64f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNormDiff_L1_64fc64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiffL1GetBufferSize_64fc64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 64-bit float complex L1 norm method on two vectors' difference, return value is 64-bit float.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiff_L1_64fc64f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNormDiff_L1_16s32s_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiffL1GetBufferSize_16s32s_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer L1 norm method on two vectors' difference, return value is 32-bit signed integer.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer..</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiff_L1_16s32s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pNorm,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNormDiff_L1_16s64s_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiffL1GetBufferSize_16s64s_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer L1 norm method on two vectors' difference, return value is 64-bit signed integer.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiff_L1_16s64s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pNorm,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNormDiff_L2_32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiffL2GetBufferSize_32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit float L2 norm method on two vectors' difference
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiff_L2_32f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNormDiff_L2_64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiffL2GetBufferSize_64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 64-bit float L2 norm method on two vectors' difference
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiff_L2_64f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNormDiff_L2_16s32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiffL2GetBufferSize_16s32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer L2 norm method on two vectors' difference, return value is 32-bit float.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiff_L2_16s32f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNormDiff_L2_32fc64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiffL2GetBufferSize_32fc64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit float complex L2 norm method on two vectors' difference, return value is 64-bit float.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiff_L2_32fc64f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNormDiff_L2_64fc64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiffL2GetBufferSize_64fc64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 64-bit float complex L2 norm method on two vectors' difference, return value is 64-bit float.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiff_L2_64fc64f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pNorm,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNormDiff_L2_16s32s_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiffL2GetBufferSize_16s32s_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer L2 norm method on two vectors' difference, return value is 32-bit signed integer.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiff_L2_16s32s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pNorm,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsNormDiff_L2Sqr_16s64s_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiffL2SqrGetBufferSize_16s64s_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer L2 Square norm method on two vectors' difference, return value is 64-bit signed integer.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pNorm">Pointer to the norm result.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsNormDiff_L2Sqr_16s64s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pNorm,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit float dot product method, return value is 32-bit float.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_32f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_32fc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_32fc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit float complex dot product method, return value is 32-bit float complex.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_32fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_32f32fc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_32f32fc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit float and 32-bit float complex dot product method, return value is 32-bit float complex.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_32f32fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_32f64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_32f64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit float dot product method, return value is 64-bit float.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_32f64f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_32fc64fc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_32fc64fc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit float complex dot product method, return value is 64-bit float complex.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_32fc64fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_32f32fc64fc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_32f32fc64fc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit float and 32-bit float complex dot product method, return value is 64-bit float complex.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_32f32fc64fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 64-bit float dot product method, return value is 64-bit float.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_64f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_64fc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_64fc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 64-bit float complex dot product method, return value is 64-bit float complex.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_64fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_64f64fc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_64f64fc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 64-bit float and 64-bit float complex dot product method, return value is 64-bit float complex.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_64f64fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_16s64s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_16s64s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer dot product method, return value is 64-bit signed integer.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_16s64s(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_16sc64sc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_16sc64sc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer complex dot product method, return value is 64-bit signed integer complex.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_16sc64sc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_16s16sc64sc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_16s16sc64sc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer and 16-bit signed short integer short dot product method, return value is 64-bit signed integer complex.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_16s16sc64sc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_16s32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_16s32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer dot product method, return value is 32-bit float.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_16s32f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_16sc32fc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_16sc32fc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer complex dot product method, return value is 32-bit float complex.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_16sc32fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_16s16sc32fc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_16s16sc32fc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer and 16-bit signed short integer complex dot product method, return value is 32-bit float complex.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_16s16sc32fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_16s_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_16s_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer dot product method, return value is 16-bit signed short integer.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_16s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_16sc_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_16sc_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer complex dot product method, return value is 16-bit signed short integer complex.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_16sc_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_32s_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_32s_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit signed integer dot product method, return value is 32-bit signed integer.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_32s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_32sc_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_32sc_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit signed integer complex dot product method, return value is 32-bit signed integer complex.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_32sc_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_16s32s_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_16s32s_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer dot product method, return value is 32-bit signed integer.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_16s32s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_16s16sc32sc_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_16s16sc32sc_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer and 16-bit signed short integer complex dot product method, return value is 32-bit signed integer complex.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_16s16sc32sc_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_16s32s32s_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_16s32s32s_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer and 32-bit signed integer dot product method, return value is 32-bit signed integer.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_16s32s32s_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_16s16sc_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_16s16sc_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer and 16-bit signed short integer complex dot product method, return value is 16-bit signed short integer complex.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_16s16sc_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_16sc32sc_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_16sc32sc_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer complex dot product method, return value is 32-bit signed integer complex.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_16sc32sc_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsDotProd_32s32sc_Sfs.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProdGetBufferSize_32s32sc_Sfs(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit signed short integer and 32-bit signed short integer complex dot product method, return value is 32-bit signed integer complex.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDp">Pointer to the dot product result.</param>
        /// <param name="nScaleFactor">integer_result_scaling.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsDotProd_32s32sc_Sfs(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDp,
            int nScaleFactor,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsCountInRange_32s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsCountInRangeGetBufferSize_32s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Computes the number of elements whose values fall into the specified range on a 32-bit signed integer array.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pCounts">Pointer to the number of elements.</param>
        /// <param name="nLowerBound">Lower bound of the specified range.</param>
        /// <param name="nUpperBound">Upper bound of the specified range.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsCountInRange_32s(
            IntPtr pSrc,
            int nLength,
            IntPtr pCounts,
            Npp32s nLowerBound,
            Npp32s nUpperBound,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsZeroCrossing_16s32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsZeroCrossingGetBufferSize_16s32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 16-bit signed short integer zero crossing method, return value is 32-bit floating point.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pValZC">Pointer to the output result.</param>
        /// <param name="tZCType">Type of the zero crossing measure: nppZCR, nppZCXor or nppZCC.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsZeroCrossing_16s32f(
            IntPtr pSrc,
            int nLength,
            IntPtr pValZC,
            NppsZCType tZCType,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsZeroCrossing_32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsZeroCrossingGetBufferSize_32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 32-bit floating-point zero crossing method, return value is 32-bit floating point.
        /// </summary>
        /// <param name="pSrc">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pValZC">Pointer to the output result.</param>
        /// <param name="tZCType">Type of the zero crossing measure: nppZCR, nppZCXor or nppZCC.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsZeroCrossing_32f(
            IntPtr pSrc,
            int nLength,
            IntPtr pValZC,
            NppsZCType tZCType,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 8-bit unsigned char maximum method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumError_8u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 8-bit signed char maximum method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumError_8s(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit unsigned short integer maximum method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumError_16u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit signed short integer maximum method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumError_16s(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit unsigned short complex integer maximum method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumError_16sc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit unsigned short integer maximum method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumError_32u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit signed short integer maximum method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumError_32s(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit unsigned short complex integer maximum method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumError_32sc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit signed short integer maximum method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumError_64s(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit unsigned short complex integer maximum method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumError_64sc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit floating point maximum method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumError_32f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit floating point complex maximum method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumError_32fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit floating point maximum method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumError_64f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit floating point complex maximum method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumError_64fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumError_8u.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumErrorGetBufferSize_8u(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumError_8s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumErrorGetBufferSize_8s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumError_16u.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumErrorGetBufferSize_16u(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumError_16s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumErrorGetBufferSize_16s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumError_16sc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumErrorGetBufferSize_16sc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumError_32u.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumErrorGetBufferSize_32u(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumError_32s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumErrorGetBufferSize_32s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumError_32sc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumErrorGetBufferSize_32sc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumError_64s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumErrorGetBufferSize_64s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumError_64sc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumErrorGetBufferSize_64sc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumError_32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumErrorGetBufferSize_32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumError_32fc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumErrorGetBufferSize_32fc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumError_64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumErrorGetBufferSize_64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumError_64fc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumErrorGetBufferSize_64fc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 8-bit unsigned char Average method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageError_8u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 8-bit signed char Average method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageError_8s(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit unsigned short integer Average method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageError_16u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit signed short integer Average method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageError_16s(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit unsigned short complex integer Average method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageError_16sc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit unsigned short integer Average method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageError_32u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit signed short integer Average method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageError_32s(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit unsigned short complex integer Average method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageError_32sc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit signed short integer Average method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageError_64s(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit unsigned short complex integer Average method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageError_64sc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit floating point Average method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageError_32f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit floating point complex Average method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageError_32fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit floating point Average method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageError_64f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit floating point complex Average method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageError_64fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageError_8u.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageErrorGetBufferSize_8u(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageError_8s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageErrorGetBufferSize_8s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageError_16u.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageErrorGetBufferSize_16u(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageError_16s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageErrorGetBufferSize_16s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageError_16sc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageErrorGetBufferSize_16sc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageError_32u.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageErrorGetBufferSize_32u(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageError_32s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageErrorGetBufferSize_32s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageError_32sc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageErrorGetBufferSize_32sc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageError_64s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageErrorGetBufferSize_64s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageError_64sc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageErrorGetBufferSize_64sc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageError_32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageErrorGetBufferSize_32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageError_32fc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageErrorGetBufferSize_32fc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageError_64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageErrorGetBufferSize_64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageError_64fc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageErrorGetBufferSize_64fc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 8-bit unsigned char MaximumRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeError_8u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 8-bit signed char MaximumRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeError_8s(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit unsigned short integer MaximumRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeError_16u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit signed short integer MaximumRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeError_16s(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit unsigned short complex integer MaximumRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeError_16sc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit unsigned short integer MaximumRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeError_32u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit signed short integer MaximumRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeError_32s(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit unsigned short complex integer MaximumRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeError_32sc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit signed short integer MaximumRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeError_64s(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit unsigned short complex integer MaximumRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeError_64sc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit floating point MaximumRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeError_32f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit floating point complex MaximumRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeError_32fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit floating point MaximumRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeError_64f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit floating point complex MaximumRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeError_64fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumRelativeError_8u.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeErrorGetBufferSize_8u(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumRelativeError_8s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeErrorGetBufferSize_8s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumRelativeError_16u.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeErrorGetBufferSize_16u(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumRelativeError_16s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeErrorGetBufferSize_16s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumRelativeError_16sc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeErrorGetBufferSize_16sc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumRelativeError_32u.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeErrorGetBufferSize_32u(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumRelativeError_32s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeErrorGetBufferSize_32s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumRelativeError_32sc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeErrorGetBufferSize_32sc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumRelativeError_64s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeErrorGetBufferSize_64s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumRelativeError_64sc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeErrorGetBufferSize_64sc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumRelativeError_32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeErrorGetBufferSize_32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumRelativeError_32fc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeErrorGetBufferSize_32fc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumRelativeError_64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeErrorGetBufferSize_64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsMaximumRelativeError_64fc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsMaximumRelativeErrorGetBufferSize_64fc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// 8-bit unsigned char AverageRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeError_8u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 8-bit signed char AverageRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeError_8s(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit unsigned short integer AverageRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeError_16u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit signed short integer AverageRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeError_16s(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 16-bit unsigned short complex integer AverageRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeError_16sc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit unsigned short integer AverageRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeError_32u(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit signed short integer AverageRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeError_32s(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit unsigned short complex integer AverageRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeError_32sc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit signed short integer AverageRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeError_64s(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit unsigned short complex integer AverageRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeError_64sc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit floating point AverageRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeError_32f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 32-bit floating point complex AverageRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeError_32fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit floating point AverageRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeError_64f(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// 64-bit floating point complex AverageRelative method.
        /// </summary>
        /// <param name="pSrc1">source_signal_pointer.</param>
        /// <param name="pSrc2">source_signal_pointer.</param>
        /// <param name="nLength">length_specification.</param>
        /// <param name="pDst">Pointer to the error result.</param>
        /// <param name="pDeviceBuffer">Pointer to the required device memory allocation, general_scratch_buffer. Use</param>
        /// <returns>signal_data_error_codes, length_error_codes.</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeError_64fc(
            IntPtr pSrc1,
            IntPtr pSrc2,
            int nLength,
            IntPtr pDst,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageRelativeError_8u.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeErrorGetBufferSize_8u(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageRelativeError_8s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeErrorGetBufferSize_8s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageRelativeError_16u.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeErrorGetBufferSize_16u(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageRelativeError_16s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeErrorGetBufferSize_16s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageRelativeError_16sc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeErrorGetBufferSize_16sc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageRelativeError_32u.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeErrorGetBufferSize_32u(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageRelativeError_32s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeErrorGetBufferSize_32s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageRelativeError_32sc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeErrorGetBufferSize_32sc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageRelativeError_64s.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeErrorGetBufferSize_64s(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageRelativeError_64sc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeErrorGetBufferSize_64sc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageRelativeError_32f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeErrorGetBufferSize_32f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageRelativeError_32fc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeErrorGetBufferSize_32fc(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageRelativeError_64f.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeErrorGetBufferSize_64f(
            int nLength,
            IntPtr hpBufferSize);

        /// <summary>
        /// Device-buffer size (in bytes) for nppsAverageRelativeError_64fc.
        /// </summary>
        /// <param name="nLength">length_specification.</param>
        /// <param name="hpBufferSize">Required buffer size. Important: hpBufferSize is a</param>
        /// <returns>NPP_SUCCESS</returns>
        [DllImport(dllFileName, SetLastError = true)]
        public static extern NppStatus nppsAverageRelativeErrorGetBufferSize_64fc(
            int nLength,
            IntPtr hpBufferSize);

    }
}