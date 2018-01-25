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
        /// 1 channel 8-bit unsigned char threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp8u nThreshold,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 8-bit unsigned char in place threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_8u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp8u nThreshold,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 16-bit unsigned short threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp16u nThreshold,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 16-bit unsigned short in place threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_16u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp16u nThreshold,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 16-bit signed short threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp16s nThreshold,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 16-bit signed short in place threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_16s_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp16s nThreshold,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 32-bit floating point threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nThreshold,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 32-bit floating point in place threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_32f_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp32f nThreshold,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 8-bit unsigned char threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 8-bit unsigned char in place threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_8u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 16-bit unsigned short threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 16-bit unsigned short in place threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_16u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 16-bit signed short threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 16-bit signed short in place threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_16s_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 32-bit floating point threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 32-bit floating point in place threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_32f_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 8-bit unsigned char image threshold, not affecting Alpha. If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha. If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_8u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 16-bit unsigned short image threshold, not affecting Alpha. If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha. If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_16u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 16-bit signed short image threshold, not affecting Alpha. If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 16-bit signed short in place image threshold, not affecting Alpha. If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_16s_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 32-bit floating point image threshold, not affecting Alpha. If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 32-bit floating point in place image threshold, not affecting Alpha. If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_32f_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 8-bit unsigned char threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp8u nThreshold);

        /// <summary>
        /// 1 channel 8-bit unsigned char in place threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_8u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp8u nThreshold);

        /// <summary>
        /// 1 channel 16-bit unsigned short threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp16u nThreshold);

        /// <summary>
        /// 1 channel 16-bit unsigned short in place threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_16u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp16u nThreshold);

        /// <summary>
        /// 1 channel 16-bit signed short threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp16s nThreshold);

        /// <summary>
        /// 1 channel 16-bit signed short in place threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_16s_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp16s nThreshold);

        /// <summary>
        /// 1 channel 32-bit floating point threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nThreshold);

        /// <summary>
        /// 1 channel 32-bit floating point in place threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_32f_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp32f nThreshold);

        /// <summary>
        /// 3 channel 8-bit unsigned char threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds);

        /// <summary>
        /// 3 channel 8-bit unsigned char in place threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_8u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds);

        /// <summary>
        /// 3 channel 16-bit unsigned short threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds);

        /// <summary>
        /// 3 channel 16-bit unsigned short in place threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_16u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds);

        /// <summary>
        /// 3 channel 16-bit signed short threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds);

        /// <summary>
        /// 3 channel 16-bit signed short in place threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_16s_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds);

        /// <summary>
        /// 3 channel 32-bit floating point threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds);

        /// <summary>
        /// 3 channel 32-bit floating point in place threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_32f_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds);

        /// <summary>
        /// 4 channel 8-bit unsigned char image threshold, not affecting Alpha. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds);

        /// <summary>
        /// 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_8u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds);

        /// <summary>
        /// 4 channel 16-bit unsigned short image threshold, not affecting Alpha. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds);

        /// <summary>
        /// 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_16u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds);

        /// <summary>
        /// 4 channel 16-bit signed short image threshold, not affecting Alpha. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds);

        /// <summary>
        /// 4 channel 16-bit signed short in place image threshold, not affecting Alpha. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_16s_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds);

        /// <summary>
        /// 4 channel 32-bit floating point image threshold, not affecting Alpha. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds);

        /// <summary>
        /// 4 channel 32-bit floating point in place image threshold, not affecting Alpha. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GT_32f_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds);

        /// <summary>
        /// 1 channel 8-bit unsigned char threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp8u nThreshold);

        /// <summary>
        /// 1 channel 8-bit unsigned char in place threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_8u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp8u nThreshold);

        /// <summary>
        /// 1 channel 16-bit unsigned short threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp16u nThreshold);

        /// <summary>
        /// 1 channel 16-bit unsigned short in place threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_16u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp16u nThreshold);

        /// <summary>
        /// 1 channel 16-bit signed short threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp16s nThreshold);

        /// <summary>
        /// 1 channel 16-bit signed short in place threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_16s_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp16s nThreshold);

        /// <summary>
        /// 1 channel 32-bit floating point threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nThreshold);

        /// <summary>
        /// 1 channel 32-bit floating point in place threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_32f_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp32f nThreshold);

        /// <summary>
        /// 3 channel 8-bit unsigned char threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds);

        /// <summary>
        /// 3 channel 8-bit unsigned char in place threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_8u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds);

        /// <summary>
        /// 3 channel 16-bit unsigned short threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds);

        /// <summary>
        /// 3 channel 16-bit unsigned short in place threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_16u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds);

        /// <summary>
        /// 3 channel 16-bit signed short threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds);

        /// <summary>
        /// 3 channel 16-bit signed short in place threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_16s_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds);

        /// <summary>
        /// 3 channel 32-bit floating point threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds);

        /// <summary>
        /// 3 channel 32-bit floating point in place threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_32f_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds);

        /// <summary>
        /// 4 channel 8-bit unsigned char image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds);

        /// <summary>
        /// 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_8u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds);

        /// <summary>
        /// 4 channel 16-bit unsigned short image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds);

        /// <summary>
        /// 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_16u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds);

        /// <summary>
        /// 4 channel 16-bit signed short image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds);

        /// <summary>
        /// 4 channel 16-bit signed short in place image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_16s_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds);

        /// <summary>
        /// 4 channel 32-bit floating point image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds);

        /// <summary>
        /// 4 channel 32-bit floating point in place image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set value is set to nThreshold, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LT_32f_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds);

        /// <summary>
        /// 1 channel 8-bit unsigned char threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp8u nThreshold,
            Npp8u nValue,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 8-bit unsigned char in place threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_8u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp8u nThreshold,
            Npp8u nValue,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 16-bit unsigned short threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp16u nThreshold,
            Npp16u nValue,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 16-bit unsigned short in place threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_16u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp16u nThreshold,
            Npp16u nValue,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 16-bit signed short threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp16s nThreshold,
            Npp16s nValue,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 16-bit signed short in place threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_16s_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp16s nThreshold,
            Npp16s nValue,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 32-bit floating point threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nThreshold,
            Npp32f nValue,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 32-bit floating point in place threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_32f_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp32f nThreshold,
            Npp32f nValue,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 8-bit unsigned char threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rValues,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 8-bit unsigned char in place threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_8u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rValues,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 16-bit unsigned short threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rValues,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 16-bit unsigned short in place threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_16u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rValues,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 16-bit signed short threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rValues,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 16-bit signed short in place threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_16s_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rValues,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 32-bit floating point threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rValues,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 32-bit floating point in place threshold. If for a comparison operations OP the predicate (sourcePixel OP nThreshold) is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_32f_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rValues,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 8-bit unsigned char image threshold, not affecting Alpha. If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel value is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rValues,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha. If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel value is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_8u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rValues,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 16-bit unsigned short image threshold, not affecting Alpha. If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel value is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rValues,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha. If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel value is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_16u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rValues,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 16-bit signed short image threshold, not affecting Alpha. If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel value is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rValues,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 16-bit signed short in place image threshold, not affecting Alpha. If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel value is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_16s_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rValues,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 32-bit floating point image threshold, not affecting Alpha. If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel value is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rValues,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 32-bit floating point in place image threshold, not affecting Alpha. If for a comparison operations OP the predicate (sourcePixel.channel OP nThreshold) is true, the channel value is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <param name="eComparisonOperation">The type of comparison operation to be used. The only valid values are: NPP_CMP_LESS and NPP_CMP_GREATER.</param>
        /// <returns>image_data_error_codes, roi_error_codes, or NPP_NOT_SUPPORTED_MODE_ERROR if an invalid comparison operation type is specified.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_Val_32f_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rValues,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 8-bit unsigned char threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp8u nThreshold,
            Npp8u nValue);

        /// <summary>
        /// 1 channel 8-bit unsigned char in place threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_8u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp8u nThreshold,
            Npp8u nValue);

        /// <summary>
        /// 1 channel 16-bit unsigned short threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp16u nThreshold,
            Npp16u nValue);

        /// <summary>
        /// 1 channel 16-bit unsigned short in place threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_16u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp16u nThreshold,
            Npp16u nValue);

        /// <summary>
        /// 1 channel 16-bit signed short threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp16s nThreshold,
            Npp16s nValue);

        /// <summary>
        /// 1 channel 16-bit signed short in place threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_16s_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp16s nThreshold,
            Npp16s nValue);

        /// <summary>
        /// 1 channel 32-bit floating point threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nThreshold,
            Npp32f nValue);

        /// <summary>
        /// 1 channel 32-bit floating point in place threshold. If for a comparison operations sourcePixel is greater than nThreshold is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement values.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_32f_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp32f nThreshold,
            Npp32f nValue);

        /// <summary>
        /// 3 channel 8-bit unsigned char threshold. If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rValues);

        /// <summary>
        /// 3 channel 8-bit unsigned char in place threshold. If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_8u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rValues);

        /// <summary>
        /// 3 channel 16-bit unsigned short threshold. If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rValues);

        /// <summary>
        /// 3 channel 16-bit unsigned short in place threshold. If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_16u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rValues);

        /// <summary>
        /// 3 channel 16-bit signed short threshold. If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rValues);

        /// <summary>
        /// 3 channel 16-bit signed short in place threshold. If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_16s_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rValues);

        /// <summary>
        /// 3 channel 32-bit floating point threshold. If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rValues);

        /// <summary>
        /// 3 channel 32-bit floating point in place threshold. If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_32f_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rValues);

        /// <summary>
        /// 4 channel 8-bit unsigned char image threshold, not affecting Alpha. If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set value is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rValues);

        /// <summary>
        /// 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha. If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set value is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_8u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rValues);

        /// <summary>
        /// 4 channel 16-bit unsigned short image threshold, not affecting Alpha. If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set value is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rValues);

        /// <summary>
        /// 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha. If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set value is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_16u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rValues);

        /// <summary>
        /// 4 channel 16-bit signed short image threshold, not affecting Alpha. If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set value is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rValues);

        /// <summary>
        /// 4 channel 16-bit signed short in place image threshold, not affecting Alpha. If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set value is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_16s_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rValues);

        /// <summary>
        /// 4 channel 32-bit floating point image threshold, not affecting Alpha. If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set value is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rValues);

        /// <summary>
        /// 4 channel 32-bit floating point in place image threshold, not affecting Alpha. If for a comparison operations sourcePixel is greater than rThreshold is true, the pixel is set value is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_GTVal_32f_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rValues);

        /// <summary>
        /// 1 channel 8-bit unsigned char threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp8u nThreshold,
            Npp8u nValue);

        /// <summary>
        /// 1 channel 8-bit unsigned char in place threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_8u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp8u nThreshold,
            Npp8u nValue);

        /// <summary>
        /// 1 channel 16-bit unsigned short threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp16u nThreshold,
            Npp16u nValue);

        /// <summary>
        /// 1 channel 16-bit unsigned short in place threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_16u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp16u nThreshold,
            Npp16u nValue);

        /// <summary>
        /// 1 channel 16-bit signed short threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp16s nThreshold,
            Npp16s nValue);

        /// <summary>
        /// 1 channel 16-bit signed short in place threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_16s_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp16s nThreshold,
            Npp16s nValue);

        /// <summary>
        /// 1 channel 32-bit floating point threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nThreshold,
            Npp32f nValue);

        /// <summary>
        /// 1 channel 32-bit floating point in place threshold. If for a comparison operations sourcePixel is less than nThreshold is true, the pixel is set to nValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThreshold">The threshold value.</param>
        /// <param name="nValue">The threshold replacement value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_32f_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp32f nThreshold,
            Npp32f nValue);

        /// <summary>
        /// 3 channel 8-bit unsigned char threshold. If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rValues);

        /// <summary>
        /// 3 channel 8-bit unsigned char in place threshold. If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_8u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rValues);

        /// <summary>
        /// 3 channel 16-bit unsigned short threshold. If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rValues);

        /// <summary>
        /// 3 channel 16-bit unsigned short in place threshold. If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_16u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rValues);

        /// <summary>
        /// 3 channel 16-bit signed short threshold. If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rValues);

        /// <summary>
        /// 3 channel 16-bit signed short in place threshold. If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_16s_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rValues);

        /// <summary>
        /// 3 channel 32-bit floating point threshold. If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rValues);

        /// <summary>
        /// 3 channel 32-bit floating point in place threshold. If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_32f_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rValues);

        /// <summary>
        /// 4 channel 8-bit unsigned char image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set value is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rValues);

        /// <summary>
        /// 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set value is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_8u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rValues);

        /// <summary>
        /// 4 channel 16-bit unsigned short image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set value is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rValues);

        /// <summary>
        /// 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set value is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_16u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rValues);

        /// <summary>
        /// 4 channel 16-bit signed short image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set value is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rValues);

        /// <summary>
        /// 4 channel 16-bit signed short in place image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set value is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_16s_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rValues);

        /// <summary>
        /// 4 channel 32-bit floating point image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set value is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rValues);

        /// <summary>
        /// 4 channel 32-bit floating point in place image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than rThreshold is true, the pixel is set value is set to rValue, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholds">The threshold values, one per color channel.</param>
        /// <param name="rValues">The threshold replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTVal_32f_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholds,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rValues);

        /// <summary>
        /// 1 channel 8-bit unsigned char threshold. If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThresholdLT">The thresholdLT value.</param>
        /// <param name="nValueLT">The thresholdLT replacement value.</param>
        /// <param name="nThresholdGT">The thresholdGT value.</param>
        /// <param name="nValueGT">The thresholdGT replacement value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp8u nThresholdLT,
            Npp8u nValueLT,
            Npp8u nThresholdGT,
            Npp8u nValueGT);

        /// <summary>
        /// 1 channel 8-bit unsigned char in place threshold. If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThresholdLT">The thresholdLT value.</param>
        /// <param name="nValueLT">The thresholdLT replacement value.</param>
        /// <param name="nThresholdGT">The thresholdGT value.</param>
        /// <param name="nValueGT">The thresholdGT replacement value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_8u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp8u nThresholdLT,
            Npp8u nValueLT,
            Npp8u nThresholdGT,
            Npp8u nValueGT);

        /// <summary>
        /// 1 channel 16-bit unsigned short threshold. If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThresholdLT">The thresholdLT value.</param>
        /// <param name="nValueLT">The thresholdLT replacement value.</param>
        /// <param name="nThresholdGT">The thresholdGT value.</param>
        /// <param name="nValueGT">The thresholdGT replacement value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp16u nThresholdLT,
            Npp16u nValueLT,
            Npp16u nThresholdGT,
            Npp16u nValueGT);

        /// <summary>
        /// 1 channel 16-bit unsigned short in place threshold. If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThresholdLT">The thresholdLT value.</param>
        /// <param name="nValueLT">The thresholdLT replacement value.</param>
        /// <param name="nThresholdGT">The thresholdGT value.</param>
        /// <param name="nValueGT">The thresholdGT replacement value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_16u_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp16u nThresholdLT,
            Npp16u nValueLT,
            Npp16u nThresholdGT,
            Npp16u nValueGT);

        /// <summary>
        /// 1 channel 16-bit signed short threshold. If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThresholdLT">The thresholdLT value.</param>
        /// <param name="nValueLT">The thresholdLT replacement value.</param>
        /// <param name="nThresholdGT">The thresholdGT value.</param>
        /// <param name="nValueGT">The thresholdGT replacement value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp16s nThresholdLT,
            Npp16s nValueLT,
            Npp16s nThresholdGT,
            Npp16s nValueGT);

        /// <summary>
        /// 1 channel 16-bit signed short in place threshold. If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThresholdLT">The thresholdLT value.</param>
        /// <param name="nValueLT">The thresholdLT replacement value.</param>
        /// <param name="nThresholdGT">The thresholdGT value.</param>
        /// <param name="nValueGT">The thresholdGT replacement value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_16s_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp16s nThresholdLT,
            Npp16s nValueLT,
            Npp16s nThresholdGT,
            Npp16s nValueGT);

        /// <summary>
        /// 1 channel 32-bit floating point threshold. If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThresholdLT">The thresholdLT value.</param>
        /// <param name="nValueLT">The thresholdLT replacement value.</param>
        /// <param name="nThresholdGT">The thresholdGT value.</param>
        /// <param name="nValueGT">The thresholdGT replacement value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nThresholdLT,
            Npp32f nValueLT,
            Npp32f nThresholdGT,
            Npp32f nValueGT);

        /// <summary>
        /// 1 channel 32-bit floating point in place threshold. If for a comparison operations sourcePixel is less than nThresholdLT is true, the pixel is set to nValueLT, else if sourcePixel is greater than nThresholdGT the pixel is set to nValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nThresholdLT">The thresholdLT value.</param>
        /// <param name="nValueLT">The thresholdLT replacement value.</param>
        /// <param name="nThresholdGT">The thresholdGT value.</param>
        /// <param name="nValueGT">The thresholdGT replacement value.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_32f_C1IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            Npp32f nThresholdLT,
            Npp32f nValueLT,
            Npp32f nThresholdGT,
            Npp32f nValueGT);

        /// <summary>
        /// 3 channel 8-bit unsigned char threshold. If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholdsLT">The thresholdLT values, one per color channel.</param>
        /// <param name="rValuesLT">The thresholdLT replacement values, one per color channel.</param>
        /// <param name="rThresholdsGT">The thresholdGT values, one per channel.</param>
        /// <param name="rValuesGT">The thresholdGT replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholdsLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rValuesLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholdsGT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rValuesGT);

        /// <summary>
        /// 3 channel 8-bit unsigned char in place threshold. If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">destination_image_pointer.</param>
        /// <param name="nSrcDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholdsLT">The thresholdLT values, one per color channel.</param>
        /// <param name="rValuesLT">The thresholdLT replacement values, one per color channel.</param>
        /// <param name="rThresholdsGT">The thresholdGT values, one per channel.</param>
        /// <param name="rValuesGT">The thresholdGT replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_8u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholdsLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rValuesLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholdsGT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rValuesGT);

        /// <summary>
        /// 3 channel 16-bit unsigned short threshold. If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholdsLT">The thresholdLT values, one per color channel.</param>
        /// <param name="rValuesLT">The thresholdLT replacement values, one per color channel.</param>
        /// <param name="rThresholdsGT">The thresholdGT values, one per channel.</param>
        /// <param name="rValuesGT">The thresholdGT replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholdsLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rValuesLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholdsGT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rValuesGT);

        /// <summary>
        /// 3 channel 16-bit unsigned short in place threshold. If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholdsLT">The thresholdLT values, one per color channel.</param>
        /// <param name="rValuesLT">The thresholdLT replacement values, one per color channel.</param>
        /// <param name="rThresholdsGT">The thresholdGT values, one per channel.</param>
        /// <param name="rValuesGT">The thresholdGT replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_16u_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholdsLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rValuesLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholdsGT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rValuesGT);

        /// <summary>
        /// 3 channel 16-bit signed short threshold. If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholdsLT">The thresholdLT values, one per color channel.</param>
        /// <param name="rValuesLT">The thresholdLT replacement values, one per color channel.</param>
        /// <param name="rThresholdsGT">The thresholdGT values, one per channel.</param>
        /// <param name="rValuesGT">The thresholdGT replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholdsLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rValuesLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholdsGT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rValuesGT);

        /// <summary>
        /// 3 channel 16-bit signed short in place threshold. If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholdsLT">The thresholdLT values, one per color channel.</param>
        /// <param name="rValuesLT">The thresholdLT replacement values, one per color channel.</param>
        /// <param name="rThresholdsGT">The thresholdGT values, one per channel.</param>
        /// <param name="rValuesGT">The thresholdGT replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_16s_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholdsLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rValuesLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholdsGT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rValuesGT);

        /// <summary>
        /// 3 channel 32-bit floating point threshold. If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholdsLT">The thresholdLT values, one per color channel.</param>
        /// <param name="rValuesLT">The thresholdLT replacement values, one per color channel.</param>
        /// <param name="rThresholdsGT">The thresholdGT values, one per channel.</param>
        /// <param name="rValuesGT">The thresholdGT replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholdsLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rValuesLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholdsGT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rValuesGT);

        /// <summary>
        /// 3 channel 32-bit floating point in place threshold. If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholdsLT">The thresholdLT values, one per color channel.</param>
        /// <param name="rValuesLT">The thresholdLT replacement values, one per color channel.</param>
        /// <param name="rThresholdsGT">The thresholdGT values, one per channel.</param>
        /// <param name="rValuesGT">The thresholdGT replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_32f_C3IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholdsLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rValuesLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholdsGT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rValuesGT);

        /// <summary>
        /// 4 channel 8-bit unsigned char image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholdsLT">The thresholdLT values, one per color channel.</param>
        /// <param name="rValuesLT">The thresholdLT replacement values, one per color channel.</param>
        /// <param name="rThresholdsGT">The thresholdGT values, one per channel.</param>
        /// <param name="rValuesGT">The thresholdGT replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholdsLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rValuesLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholdsGT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rValuesGT);

        /// <summary>
        /// 4 channel 8-bit unsigned char in place image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholdsLT">The thresholdLT values, one per color channel.</param>
        /// <param name="rValuesLT">The thresholdLT replacement values, one per color channel.</param>
        /// <param name="rThresholdsGT">The thresholdGT values, one per channel.</param>
        /// <param name="rValuesGT">The thresholdGT replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_8u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholdsLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rValuesLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rThresholdsGT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp8u[] rValuesGT);

        /// <summary>
        /// 4 channel 16-bit unsigned short image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholdsLT">The thresholdLT values, one per color channel.</param>
        /// <param name="rValuesLT">The thresholdLT replacement values, one per color channel.</param>
        /// <param name="rThresholdsGT">The thresholdGT values, one per channel.</param>
        /// <param name="rValuesGT">The thresholdGT replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholdsLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rValuesLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholdsGT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rValuesGT);

        /// <summary>
        /// 4 channel 16-bit unsigned short in place image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholdsLT">The thresholdLT values, one per color channel.</param>
        /// <param name="rValuesLT">The thresholdLT replacement values, one per color channel.</param>
        /// <param name="rThresholdsGT">The thresholdGT values, one per channel.</param>
        /// <param name="rValuesGT">The thresholdGT replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_16u_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholdsLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rValuesLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rThresholdsGT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16u[] rValuesGT);

        /// <summary>
        /// 4 channel 16-bit signed short image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholdsLT">The thresholdLT values, one per color channel.</param>
        /// <param name="rValuesLT">The thresholdLT replacement values, one per color channel.</param>
        /// <param name="rThresholdsGT">The thresholdGT values, one per channel.</param>
        /// <param name="rValuesGT">The thresholdGT replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholdsLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rValuesLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholdsGT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rValuesGT);

        /// <summary>
        /// 4 channel 16-bit signed short in place image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholdsLT">The thresholdLT values, one per color channel.</param>
        /// <param name="rValuesLT">The thresholdLT replacement values, one per color channel.</param>
        /// <param name="rThresholdsGT">The thresholdGT values, one per channel.</param>
        /// <param name="rValuesGT">The thresholdGT replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_16s_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholdsLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rValuesLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rThresholdsGT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp16s[] rValuesGT);

        /// <summary>
        /// 4 channel 32-bit floating point image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholdsLT">The thresholdLT values, one per color channel.</param>
        /// <param name="rValuesLT">The thresholdLT replacement values, one per color channel.</param>
        /// <param name="rThresholdsGT">The thresholdGT values, one per channel.</param>
        /// <param name="rValuesGT">The thresholdGT replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholdsLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rValuesLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholdsGT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rValuesGT);

        /// <summary>
        /// 4 channel 32-bit floating point in place image threshold, not affecting Alpha. If for a comparison operations sourcePixel is less than rThresholdLT is true, the pixel is set value is set to rValueLT, else if sourcePixel is greater than rThresholdGT the pixel is set to rValueGT, otherwise it is set to sourcePixel.
        /// </summary>
        /// <param name="pSrcDst">in_place_image_pointer.</param>
        /// <param name="nSrcDstStep">in_place_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="rThresholdsLT">The thresholdLT values, one per color channel.</param>
        /// <param name="rValuesLT">The thresholdLT replacement values, one per color channel.</param>
        /// <param name="rThresholdsGT">The thresholdGT values, one per channel.</param>
        /// <param name="rValuesGT">The thresholdGT replacement values, one per color channel.</param>
        /// <returns>image_data_error_codes, roi_error_codes.</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiThreshold_LTValGTVal_32f_AC4IR(
            IntPtr pSrcDst,
            int nSrcDstStep,
            NppiSize oSizeROI,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholdsLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rValuesLT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rThresholdsGT,
            [MarshalAs(UnmanagedType.LPArray, SizeConst = 3)]Npp32f[] rValuesGT);

        /// <summary>
        /// 1 channel 8-bit unsigned char image compare. Compare pSrc1's pixels with corresponding pixels in pSrc2.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompare_8u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 8-bit unsigned char image compare. Compare pSrc1's pixels with corresponding pixels in pSrc2.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompare_8u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 8-bit unsigned char image compare. Compare pSrc1's pixels with corresponding pixels in pSrc2.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompare_8u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 8-bit unsigned char image compare, not affecting Alpha. Compare pSrc1's pixels with corresponding pixels in pSrc2.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompare_8u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 16-bit unsigned short image compare. Compare pSrc1's pixels with corresponding pixels in pSrc2.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompare_16u_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 16-bit unsigned short image compare. Compare pSrc1's pixels with corresponding pixels in pSrc2.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompare_16u_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 16-bit unsigned short image compare. Compare pSrc1's pixels with corresponding pixels in pSrc2.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompare_16u_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 16-bit unsigned short image compare, not affecting Alpha. Compare pSrc1's pixels with corresponding pixels in pSrc2.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompare_16u_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 16-bit signed short image compare. Compare pSrc1's pixels with corresponding pixels in pSrc2.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompare_16s_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 16-bit signed short image compare. Compare pSrc1's pixels with corresponding pixels in pSrc2.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompare_16s_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 16-bit signed short image compare. Compare pSrc1's pixels with corresponding pixels in pSrc2.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompare_16s_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 16-bit signed short image compare, not affecting Alpha. Compare pSrc1's pixels with corresponding pixels in pSrc2.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompare_16s_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 32-bit floating point image compare. Compare pSrc1's pixels with corresponding pixels in pSrc2.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompare_32f_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 32-bit floating point image compare. Compare pSrc1's pixels with corresponding pixels in pSrc2.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompare_32f_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 32-bit floating point image compare. Compare pSrc1's pixels with corresponding pixels in pSrc2.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompare_32f_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 32-bit signed floating point compare, not affecting Alpha. Compare pSrc1's pixels with corresponding pixels in pSrc2.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompare_32f_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 8-bit unsigned char image compare with constant value. Compare pSrc's pixels with constant value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="nConstant">constant value.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareC_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            Npp8u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 8-bit unsigned char image compare with constant value. Compare pSrc's pixels with constant value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pConstants">pointer to a list of constant values, one per color channel..</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareC_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 8-bit unsigned char image compare with constant value. Compare pSrc's pixels with constant value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pConstants">pointer to a list of constants, one per color channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareC_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 8-bit unsigned char image compare, not affecting Alpha. Compare pSrc's pixels with constant value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pConstants">pointer to a list of constants, one per color channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareC_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 16-bit unsigned short image compare with constant value. Compare pSrc's pixels with constant value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="nConstant">constant value</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareC_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            Npp16u nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 16-bit unsigned short image compare with constant value. Compare pSrc's pixels with constant value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pConstants">pointer to a list of constants, one per color channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareC_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 16-bit unsigned short image compare with constant value. Compare pSrc's pixels with constant value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pConstants">pointer to a list of constants, one per color channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareC_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 16-bit unsigned short image compare, not affecting Alpha. Compare pSrc's pixels with constant value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pConstants">pointer to a list of constants, one per color channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareC_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 16-bit signed short image compare with constant value. Compare pSrc's pixels with constant value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="nConstant">constant value.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareC_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            Npp16s nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 16-bit signed short image compare with constant value. Compare pSrc's pixels with constant value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pConstants">pointer to a list of constants, one per color channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareC_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 16-bit signed short image compare with constant value. Compare pSrc's pixels with constant value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pConstants">pointer to a list of constants, one per color channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareC_16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 16-bit signed short image compare, not affecting Alpha. Compare pSrc's pixels with constant value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pConstants">pointer to a list of constants, one per color channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareC_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 32-bit floating point image compare with constant value. Compare pSrc's pixels with constant value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="nConstant">constant value</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareC_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            Npp32f nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 3 channel 32-bit floating point image compare with constant value. Compare pSrc's pixels with constant value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pConstants">pointer to a list of constants, one per color channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareC_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 32-bit floating point image compare with constant value. Compare pSrc's pixels with constant value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pConstants">pointer to a list of constants, one per color channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareC_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 4 channel 32-bit signed floating point compare, not affecting Alpha. Compare pSrc's pixels with constant value.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pConstants">pointer to a list of constants, one per color channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eComparisonOperation">Specifies the comparison operation to be used in the pixel comparison.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareC_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            NppCmpOp eComparisonOperation);

        /// <summary>
        /// 1 channel 32-bit floating point image compare whether two images are equal within epsilon. Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nEpsilon">epsilon tolerance value to compare to pixel absolute differences</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareEqualEps_32f_C1R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nEpsilon);

        /// <summary>
        /// 3 channel 32-bit floating point image compare whether two images are equal within epsilon. Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nEpsilon">epsilon tolerance value to compare to per color channel pixel absolute differences</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareEqualEps_32f_C3R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nEpsilon);

        /// <summary>
        /// 4 channel 32-bit floating point image compare whether two images are equal within epsilon. Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nEpsilon">epsilon tolerance value to compare to per color channel pixel absolute differences</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareEqualEps_32f_C4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nEpsilon);

        /// <summary>
        /// 4 channel 32-bit signed floating point compare whether two images are equal within epsilon, not affecting Alpha. Compare pSrc1's pixels with corresponding pixels in pSrc2 to determine whether they are equal with a difference of epsilon.
        /// </summary>
        /// <param name="pSrc1">source_image_pointer.</param>
        /// <param name="nSrc1Step">source_image_line_step.</param>
        /// <param name="pSrc2">source_image_pointer.</param>
        /// <param name="nSrc2Step">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nEpsilon">epsilon tolerance value to compare to per color channel pixel absolute differences</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareEqualEps_32f_AC4R(
            IntPtr pSrc1,
            int nSrc1Step,
            IntPtr pSrc2,
            int nSrc2Step,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nEpsilon);

        /// <summary>
        /// 1 channel 32-bit floating point image compare whether image and constant are equal within epsilon. Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="nConstant">constant value</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nEpsilon">epsilon tolerance value to compare to pixel absolute differences</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareEqualEpsC_32f_C1R(
            IntPtr pSrc,
            int nSrcStep,
            Npp32f nConstant,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nEpsilon);

        /// <summary>
        /// 3 channel 32-bit floating point image compare whether image and constant are equal within epsilon. Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pConstants">pointer to a list of constants, one per color channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nEpsilon">epsilon tolerance value to compare to per color channel pixel absolute differences</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareEqualEpsC_32f_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nEpsilon);

        /// <summary>
        /// 4 channel 32-bit floating point image compare whether image and constant are equal within epsilon. Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pConstants">pointer to a list of constants, one per color channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nEpsilon">epsilon tolerance value to compare to per color channel pixel absolute differences</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareEqualEpsC_32f_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nEpsilon);

        /// <summary>
        /// 4 channel 32-bit signed floating point compare whether image and constant are equal within epsilon, not affecting Alpha. Compare pSrc's pixels with constant value to determine whether they are equal within a difference of epsilon.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pConstants">pointer to a list of constants, one per color channel.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nEpsilon">epsilon tolerance value to compare to per color channel pixel absolute differences</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(tcDll, SetLastError = true)]
        public static extern NppStatus nppiCompareEqualEpsC_32f_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pConstants,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            Npp32f nEpsilon);

    }
}