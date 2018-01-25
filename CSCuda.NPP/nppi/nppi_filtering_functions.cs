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
        /// 8-bit unsigned single-channel 1D column convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 8-bit unsigned three-channel 1D column convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 8-bit unsigned four-channel 1D column convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 8-bit unsigned four-channel 1D column convolution ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 16-bit unsigned single-channel 1D column convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 16-bit unsigned three-channel 1D column convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 16-bit unsigned four-channel 1D column convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 16-bit unsigned four-channel 1D column convolution ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 16-bit single-channel 1D column convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 16-bit three-channel 1D column convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 16-bit four-channel 1D column convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 16-bit four-channel 1D column convolution ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 32-bit float single-channel 1D column convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 32-bit float three-channel 1D column convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 32-bit float four-channel 1D column convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 32-bit float four-channel 1D column convolution ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 64-bit float single-channel 1D column convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn_64f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// Single channel 8-bit unsigned 1D column convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned 1D column convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel channel 8-bit unsigned 1D column convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned convolution 1D column filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit unsigned convolution 1D column filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit unsigned 1D column convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel channel 16-bit 1D column unsigned convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned 1D column convolution filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit 1D column convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit 1D column convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel channel 16-bit 1D column convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit 1D column convolution filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit float 1D column convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 32-bit float 1D column convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit float 1D column convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit float 1D column convolution filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// 8-bit unsigned single-channel 1D column convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn32f_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 8-bit unsigned three-channel 1D column convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn32f_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 8-bit unsigned four-channel 1D column convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn32f_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 8-bit unsigned four-channel 1D column convolution ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn32f_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 16-bit unsigned single-channel 1D column convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn32f_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 16-bit unsigned three-channel 1D column convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn32f_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 16-bit unsigned four-channel 1D column convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn32f_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 16-bit unsigned four-channel 1D column convolution ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn32f_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 16-bit single-channel 1D column convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn32f_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 16-bit three-channel 1D column convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn32f_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 16-bit four-channel 1D column convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn32f_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 16-bit four-channel 1D column convolution ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumn32f_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// Single channel 8-bit unsigned 1D column convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder32f_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned 1D column convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder32f_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned 1D column convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder32f_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned 1D column convolution filter with border control, ignorint alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder32f_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit unsigned 1D column convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder32f_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit unsigned 1D column convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder32f_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned 1D column convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder32f_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned 1D column convolution filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder32f_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit 1D column convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder32f_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit 1D column convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder32f_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit 1D column convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder32f_16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit 1D column convolution filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterColumnBorder32f_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// 8-bit unsigned single-channel 1D row convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 8-bit unsigned three-channel 1D row convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 8-bit unsigned four-channel 1D row convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 8-bit unsigned four-channel 1D row convolution ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 16-bit unsigned single-channel 1D row convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 16-bit unsigned three-channel 1D row convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 16-bit unsigned four-channel 1D row convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 16-bit unsigned four-channel 1D row convolution ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 16-bit single-channel 1D row convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 16-bit three-channel 1D row convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 16-bit four-channel 1D row convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 16-bit four-channel 1D row convolution ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// 32-bit float single-channel 1D row convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 32-bit float three-channel 1D row convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 32-bit float four-channel 1D row convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 32-bit float four-channel 1D row convolution ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 64-bit float single-channel 1D row convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow_64f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// Single channel 8-bit unsigned 1D row convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned 1D row convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel channel 8-bit unsigned 1D row convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned convolution 1D row filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit unsigned convolution 1D row filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit unsigned 1D row convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel channel 16-bit 1D row unsigned convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned 1D row convolution filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit 1D row convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit 1D row convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel channel 16-bit 1D row convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit 1D row convolution filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit float 1D row convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 32-bit float 1D row convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit float 1D row convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit float 1D row convolution filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// 8-bit unsigned single-channel 1D row convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow32f_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 8-bit unsigned three-channel 1D row convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow32f_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 8-bit unsigned four-channel 1D row convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow32f_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 8-bit unsigned four-channel 1D row convolution ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow32f_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 16-bit unsigned single-channel 1D row convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow32f_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 16-bit unsigned three-channel 1D row convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow32f_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 16-bit unsigned four-channel 1D row convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow32f_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 16-bit unsigned four-channel 1D row convolution ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow32f_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 16-bit single-channel 1D row convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow32f_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 16-bit three-channel 1D row convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow32f_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 16-bit four-channel 1D row convolution.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow32f_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// 16-bit four-channel 1D row convolution ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coefficients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRow32f_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// Single channel 8-bit unsigned 1D row convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder32f_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned 1D row convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder32f_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned 1D row convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder32f_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned 1D row convolution filter with border control, ignorint alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder32f_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit unsigned 1D row convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder32f_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit unsigned 1D row convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder32f_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned 1D row convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder32f_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned 1D row convolution filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder32f_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit 1D row convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder32f_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit 1D row convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder32f_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit 1D row convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder32f_16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit 1D row convolution filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="nMaskSize">Width of the kernel.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRowBorder32f_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// One channel 8-bit unsigned 1D (column) sum to 32f.
        /// Apply Column Window Summation filter over a 1D mask region around each source pixel for 1-channel 8 bit/pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring column pixel values in a mask region of the source image defined by nMaskSize and nAnchor.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowColumn_8u32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// Three channel 8-bit unsigned 1D (column) sum to 32f.
        /// Apply Column Window Summation filter over a 1D mask region around each source pixel for 3-channel 8 bit/pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring column pixel values in a mask region of the source image defined by nMaskSize and nAnchor.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowColumn_8u32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// Four channel 8-bit unsigned 1D (column) sum to 32f.
        /// Apply Column Window Summation filter over a 1D mask region around each source pixel for 4-channel 8 bit/pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring column pixel values in a mask region of the source image defined by nMaskSize and nAnchor.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowColumn_8u32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// One channel 16-bit unsigned 1D (column) sum to 32f.
        /// Apply Column Window Summation filter over a 1D mask region around each source pixel for 1-channel 16 bit/pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring column pixel values in a mask region of the source image defined by nMaskSize and nAnchor.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowColumn_16u32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// Three channel 16-bit unsigned 1D (column) sum to 32f.
        /// Apply Column Window Summation filter over a 1D mask region around each source pixel for 3-channel 16 bit/pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring column pixel values in a mask region of the source image defined by nMaskSize and nAnchor.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowColumn_16u32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// Four channel 16-bit unsigned 1D (column) sum to 32f.
        /// Apply Column Window Summation filter over a 1D mask region around each source pixel for 4-channel 16 bit/pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring column pixel values in a mask region of the source image defined by nMaskSize and nAnchor.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowColumn_16u32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// One channel 16-bit signed 1D (column) sum to 32f.
        /// Apply Column Window Summation filter over a 1D mask region around each source pixel for 1-channel 16 bit/pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring column pixel values in a mask region of the source image defined by nMaskSize and nAnchor.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowColumn_16s32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// Three channel 16-bit signed 1D (column) sum to 32f.
        /// Apply Column Window Summation filter over a 1D mask region around each source pixel for 1-channel 16 bit/pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring column pixel values in a mask region of the source image defined by nMaskSize and nAnchor.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowColumn_16s32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// Four channel 16-bit signed 1D (column) sum to 32f.
        /// Apply Column Window Summation filter over a 1D mask region around each source pixel for 4-channel 16 bit/pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring column pixel values in a mask region of the source image defined by nMaskSize and nAnchor.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowColumn_16s32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// One channel 8-bit unsigned 1D (row) sum to 32f.
        /// Apply Row Window Summation filter over a 1D mask region around each source pixel for 1-channel 8-bit pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring row pixel values in a mask region of the source image defined by iKernelDim and iAnchorX.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowRow_8u32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// Three channel 8-bit unsigned 1D (row) sum to 32f.
        /// Apply Row Window Summation filter over a 1D mask region around each source pixel for 3-channel 8-bit pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring row pixel values in a mask region of the source image defined by iKernelDim and iAnchorX.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowRow_8u32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// Four channel 8-bit unsigned 1D (row) sum to 32f.
        /// Apply Row Window Summation filter over a 1D mask region around each source pixel for 4-channel 8-bit pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring row pixel values in a mask region of the source image defined by iKernelDim and iAnchorX.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowRow_8u32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// One channel 16-bit unsigned 1D (row) sum to 32f.
        /// Apply Row Window Summation filter over a 1D mask region around each source pixel for 1-channel 16-bit pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring row pixel values in a mask region of the source image defined by iKernelDim and iAnchorX.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowRow_16u32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// Three channel 16-bit unsigned 1D (row) sum to 32f.
        /// Apply Row Window Summation filter over a 1D mask region around each source pixel for 3-channel 16-bit pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring row pixel values in a mask region of the source image defined by iKernelDim and iAnchorX.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowRow_16u32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// Four channel 16-bit unsigned 1D (row) sum to 32f.
        /// Apply Row Window Summation filter over a 1D mask region around each source pixel for 4-channel 16-bit pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring row pixel values in a mask region of the source image defined by iKernelDim and iAnchorX.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowRow_16u32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// One channel 16-bit signed 1D (row) sum to 32f.
        /// Apply Row Window Summation filter over a 1D mask region around each source pixel for 1-channel 16-bit pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring row pixel values in a mask region of the source image defined by iKernelDim and iAnchorX.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowRow_16s32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// Three channel 16-bit signed 1D (row) sum to 32f.
        /// Apply Row Window Summation filter over a 1D mask region around each source pixel for 3-channel 16-bit pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring row pixel values in a mask region of the source image defined by iKernelDim and iAnchorX.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowRow_16s32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// Four channel 16-bit signed 1D (row) sum to 32f.
        /// Apply Row Window Summation filter over a 1D mask region around each source pixel for 4-channel 16-bit pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring row pixel values in a mask region of the source image defined by iKernelDim and iAnchorX.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowRow_16s32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor);

        /// <summary>
        /// One channel 8-bit unsigned 1D (column) sum to 32f with border control.
        /// Apply Column Window Summation filter over a 1D mask region around each source pixel for 1-channel 8 bit/pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring column pixel values in a mask region of the source image defined by nMaskSize and nAnchor.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowColumnBorder_8u32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned 1D (column) sum to 32f with border control.
        /// Apply Column Window Summation filter over a 1D mask region around each source pixel for 3-channel 8 bit/pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring column pixel values in a mask region of the source image defined by nMaskSize and nAnchor.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowColumnBorder_8u32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned 1D (column) sum to 32f with border control.
        /// Apply Column Window Summation filter over a 1D mask region around each source pixel for 4-channel 8 bit/pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring column pixel values in a mask region of the source image defined by nMaskSize and nAnchor.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowColumnBorder_8u32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// One channel 16-bit unsigned 1D (column) sum to 32f with border control.
        /// Apply Column Window Summation filter over a 1D mask region around each source pixel for 1-channel 16 bit/pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring column pixel values in a mask region of the source image defined by nMaskSize and nAnchor.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowColumnBorder_16u32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit unsigned 1D (column) sum to 32f with border control.
        /// Apply Column Window Summation filter over a 1D mask region around each source pixel for 3-channel 16 bit/pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring column pixel values in a mask region of the source image defined by nMaskSize and nAnchor.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowColumnBorder_16u32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned 1D (column) sum to 32f with border control.
        /// Apply Column Window Summation filter over a 1D mask region around each source pixel for 4-channel 16 bit/pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring column pixel values in a mask region of the source image defined by nMaskSize and nAnchor.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowColumnBorder_16u32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// One channel 16-bit signed 1D (column) sum to 32f with border control.
        /// Apply Column Window Summation filter over a 1D mask region around each source pixel for 1-channel 16 bit/pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring column pixel values in a mask region of the source image defined by nMaskSize and nAnchor.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowColumnBorder_16s32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit signed 1D (column) sum to 32f with border control.
        /// Apply Column Window Summation filter over a 1D mask region around each source pixel for 1-channel 16 bit/pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring column pixel values in a mask region of the source image defined by nMaskSize and nAnchor.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowColumnBorder_16s32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed 1D (column) sum to 32f with border control.
        /// Apply Column Window Summation filter over a 1D mask region around each source pixel for 4-channel 16 bit/pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring column pixel values in a mask region of the source image defined by nMaskSize and nAnchor.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">Y offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowColumnBorder_16s32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// One channel 8-bit unsigned 1D (row) sum to 32f with border control.
        /// Apply Row Window Summation filter over a 1D mask region around each source pixel for 1-channel 8-bit pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring row pixel values in a mask region of the source image defined by iKernelDim and iAnchorX.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowRowBorder_8u32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned 1D (row) sum to 32f with border control.
        /// Apply Row Window Summation filter over a 1D mask region around each source pixel for 3-channel 8-bit pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring row pixel values in a mask region of the source image defined by iKernelDim and iAnchorX.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowRowBorder_8u32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned 1D (row) sum to 32f with border control.
        /// Apply Row Window Summation filter over a 1D mask region around each source pixel for 4-channel 8-bit pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring row pixel values in a mask region of the source image defined by iKernelDim and iAnchorX.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowRowBorder_8u32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// One channel 16-bit unsigned 1D (row) sum to 32f with border control.
        /// Apply Row Window Summation filter over a 1D mask region around each source pixel for 1-channel 16-bit pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring row pixel values in a mask region of the source image defined by iKernelDim and iAnchorX.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowRowBorder_16u32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit unsigned 1D (row) sum to 32f with border control.
        /// Apply Row Window Summation filter over a 1D mask region around each source pixel for 3-channel 16-bit pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring row pixel values in a mask region of the source image defined by iKernelDim and iAnchorX.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowRowBorder_16u32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned 1D (row) sum to 32f with border control.
        /// Apply Row Window Summation filter over a 1D mask region around each source pixel for 4-channel 16-bit pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring row pixel values in a mask region of the source image defined by iKernelDim and iAnchorX.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowRowBorder_16u32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// One channel 16-bit signed 1D (row) sum to 32f with border control.
        /// Apply Row Window Summation filter over a 1D mask region around each source pixel for 1-channel 16-bit pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring row pixel values in a mask region of the source image defined by iKernelDim and iAnchorX.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowRowBorder_16s32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit signed 1D (row) sum to 32f with border control.
        /// Apply Row Window Summation filter over a 1D mask region around each source pixel for 3-channel 16-bit pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring row pixel values in a mask region of the source image defined by iKernelDim and iAnchorX.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowRowBorder_16s32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed 1D (row) sum to 32f with border control.
        /// Apply Row Window Summation filter over a 1D mask region around each source pixel for 4-channel 16-bit pixel input images with 32-bit floating point output. Result 32-bit floating point pixel is equal to the sum of the corresponding and neighboring row pixel values in a mask region of the source image defined by iKernelDim and iAnchorX.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oROI">roi_specification.</param>
        /// <param name="nMaskSize">Length of the linear kernel array.</param>
        /// <param name="nAnchor">X offset of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiSumWindowRowBorder_16s32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oROI,
            Npp32s nMaskSize,
            Npp32s nAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// Three channel 8-bit unsigned convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// Four channel channel 8-bit unsigned convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// Four channel 8-bit unsigned convolution filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// Single channel 16-bit unsigned convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// Three channel 16-bit unsigned convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// Four channel channel 16-bit unsigned convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// Four channel 16-bit unsigned convolution filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// Single channel 16-bit convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// Three channel 16-bit convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// Four channel channel 16-bit convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// Four channel 16-bit convolution filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor);

        /// <summary>
        /// Single channel 32-bit float convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Two channel 32-bit float convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter_32f_C2R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three channel 32-bit float convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 32-bit float convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 32-bit float convolution filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single channel 64-bit float convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter_64f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single channel 8-bit unsigned convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Two channel 8-bit unsigned convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_8u_C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three channel 8-bit unsigned convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 8-bit unsigned convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 8-bit unsigned convolution filter, ignorint alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single channel 8-bit signed convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_8s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Two channel 8-bit signed convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_8s_C2R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three channel 8-bit signed convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_8s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 8-bit signed convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_8s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 8-bit signed convolution filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_8s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single channel 16-bit unsigned convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three channel 16-bit unsigned convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 16-bit unsigned convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 16-bit unsigned convolution filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single channel 16-bit convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three channel 16-bit convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 16-bit convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 16-bit convolution filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single channel 32-bit convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_32s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three channel 32-bit convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_32s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 32-bit convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_32s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 32-bit convolution filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_32s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single channel 8-bit unsigned to 16-bit signed convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_8u16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three channel 8-bit unsigned to 16-bit signed convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_8u16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 8-bit unsigned to 16-bit signed convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_8u16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 8-bit unsigned to 16-bit signed convolution filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_8u16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single channel 8-bit to 16-bit signed convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_8s16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three channel 8-bit to 16-bit signed convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_8s16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 8-bit to 16-bit signed convolution filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_8s16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 8-bit to 16-bit signed convolution filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilter32f_8s16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single channel 8-bit unsigned convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel channel 8-bit unsigned convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned convolution filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit unsigned convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit unsigned convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel channel 16-bit unsigned convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned convolution filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel channel 16-bit convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit convolution filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="nDivisor">The factor by which the convolved summation from the Filter operation should be divided. If equal to the sum of coefficients, this will keep the maximum result value within full scale.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            Npp32s nDivisor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit float convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Two channel 32-bit float convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder_32f_C2R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 32-bit float convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit float convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit float convolution filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_8u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Two channel 8-bit unsigned convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_8u_C2R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_8u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_8u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned convolution filter with border control, ignorint alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_8u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit signed convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_8s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Two channel 8-bit signed convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_8s_C2R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit signed convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_8s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit signed convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_8s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit signed convolution filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_8s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit unsigned convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_16u_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit unsigned convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_16u_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_16u_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned convolution filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_16u_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit convolution filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_32s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 32-bit convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_32s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_32s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit convolution filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_32s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned to 16-bit signed convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_8u16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned to 16-bit signed convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_8u16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned to 16-bit signed convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_8u16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned to 16-bit signed convolution filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_8u16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit to 16-bit signed convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_8s16s_C1R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit to 16-bit signed convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_8s16s_C3R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit to 16-bit signed convolution filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_8s16s_C4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit to 16-bit signed convolution filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="pKernel">Pointer to the start address of the kernel coefficient array. Coeffcients are expected to be stored in reverse order.</param>
        /// <param name="oKernelSize">Width and Height of the rectangular kernel.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBorder32f_8s16s_AC4R(
            IntPtr pSrc,
            int nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            int nDstStep,
            NppiSize oSizeROI,
            IntPtr pKernel,
            NppiSize oKernelSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned box filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBox_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three channel 8-bit unsigned box filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBox_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 8-bit unsigned box filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBox_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 8-bit unsigned box filter, ignorting alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBox_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single channel 16-bit unsigned box filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBox_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three channel 16-bit unsigned box filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBox_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 16-bit unsigned box filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBox_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 16-bit unsigned box filter, ignorting alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBox_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single channel 16-bit box filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBox_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three channel 16-bit box filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBox_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 16-bit box filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBox_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 16-bit box filter, ignorting alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBox_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single channel 32-bit floating-point box filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBox_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three channel 32-bit floating-point box filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBox_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 32-bit floating-point box filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBox_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 32-bit floating-point box filter, ignorting alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBox_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single channel 64-bit floating-point box filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBox_64f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single channel 8-bit unsigned box filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBoxBorder_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned box filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBoxBorder_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned box filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBoxBorder_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned box filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBoxBorder_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit unsigned box filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBoxBorder_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit unsigned box filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBoxBorder_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned box filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBoxBorder_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned box filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBoxBorder_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit box filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBoxBorder_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit box filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBoxBorder_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit box filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBoxBorder_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit box filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBoxBorder_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit floating-point box filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBoxBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 32-bit floating-point box filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBoxBorder_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point box filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBoxBorder_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point box filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Avg operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterBoxBorder_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned maximum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMax_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three channel 8-bit unsigned maximum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMax_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 8-bit unsigned maximum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMax_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 8-bit unsigned maximum filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMax_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single channel 16-bit unsigned maximum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMax_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three channel 16-bit unsigned maximum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMax_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 16-bit unsigned maximum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMax_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 16-bit unsigned maximum filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMax_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single channel 16-bit signed maximum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMax_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three channel 16-bit signed maximum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMax_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 16-bit signed maximum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMax_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 16-bit signed maximum filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMax_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single channel 32-bit floating-point maximum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMax_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three channel 32-bit floating-point maximum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMax_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 32-bit floating-point maximum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMax_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 32-bit floating-point maximum filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMax_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single channel 8-bit unsigned maximum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMaxBorder_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned maximum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMaxBorder_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned maximum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMaxBorder_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned maximum filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMaxBorder_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit unsigned maximum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMaxBorder_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit unsigned maximum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMaxBorder_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned maximum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMaxBorder_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned maximum filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMaxBorder_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit signed maximum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMaxBorder_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit signed maximum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMaxBorder_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed maximum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMaxBorder_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed maximum filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMaxBorder_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit floating-point maximum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMaxBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 32-bit floating-point maximum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMaxBorder_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point maximum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMaxBorder_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point maximum filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMaxBorder_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned minimum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMin_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three channel 8-bit unsigned minimum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMin_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 8-bit unsigned minimum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMin_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 8-bit unsigned minimum filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMin_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single channel 16-bit unsigned minimum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMin_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three channel 16-bit unsigned minimum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMin_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 16-bit unsigned minimum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMin_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 16-bit unsigned minimum filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMin_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single channel 16-bit signed minimum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMin_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three channel 16-bit signed minimum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMin_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 16-bit signed minimum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMin_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 16-bit signed minimum filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMin_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single channel 32-bit floating-point minimum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMin_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Three channel 32-bit floating-point minimum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMin_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 32-bit floating-point minimum filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMin_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Four channel 32-bit floating-point minimum filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Max operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMin_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor);

        /// <summary>
        /// Single channel 8-bit unsigned minimum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Min operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMinBorder_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned minimum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Min operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMinBorder_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned minimum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Min operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMinBorder_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned minimum filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Min operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMinBorder_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit unsigned minimum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Min operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMinBorder_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit unsigned minimum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Min operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMinBorder_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned minimum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Min operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMinBorder_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned minimum filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Min operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMinBorder_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit signed minimum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Min operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMinBorder_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit signed minimum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Min operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMinBorder_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed minimum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Min operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMinBorder_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed minimum filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Min operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMinBorder_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit floating-point minimum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Min operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMinBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 32-bit floating-point minimum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Min operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMinBorder_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point minimum filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Min operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMinBorder_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point minimum filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Min operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMinBorder_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned median filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="pBuffer">Pointer to the user-allocated scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedian_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            IntPtr pBuffer);

        /// <summary>
        /// Three channel 8-bit unsigned median filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="pBuffer">Pointer to the user-allocated scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedian_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            IntPtr pBuffer);

        /// <summary>
        /// Four channel 8-bit unsigned median filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="pBuffer">Pointer to the user-allocated scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedian_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            IntPtr pBuffer);

        /// <summary>
        /// Four channel 8-bit unsigned median filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="pBuffer">Pointer to the user-allocated scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedian_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            IntPtr pBuffer);

        /// <summary>
        /// Single channel 16-bit unsigned median filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="pBuffer">Pointer to the user-allocated scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedian_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            IntPtr pBuffer);

        /// <summary>
        /// Three channel 16-bit unsigned median filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="pBuffer">Pointer to the user-allocated scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedian_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            IntPtr pBuffer);

        /// <summary>
        /// Four channel 16-bit unsigned median filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="pBuffer">Pointer to the user-allocated scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedian_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            IntPtr pBuffer);

        /// <summary>
        /// Four channel 16-bit unsigned median filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="pBuffer">Pointer to the user-allocated scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedian_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            IntPtr pBuffer);

        /// <summary>
        /// Single channel 16-bit signed median filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="pBuffer">Pointer to the user-allocated scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedian_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            IntPtr pBuffer);

        /// <summary>
        /// Three channel 16-bit signed median filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="pBuffer">Pointer to the user-allocated scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedian_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            IntPtr pBuffer);

        /// <summary>
        /// Four channel 16-bit signed median filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="pBuffer">Pointer to the user-allocated scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedian_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            IntPtr pBuffer);

        /// <summary>
        /// Four channel 16-bit signed median filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="pBuffer">Pointer to the user-allocated scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedian_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            IntPtr pBuffer);

        /// <summary>
        /// Single channel 32-bit floating-point median filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="pBuffer">Pointer to the user-allocated scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedian_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            IntPtr pBuffer);

        /// <summary>
        /// Three channel 32-bit floating-point median filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="pBuffer">Pointer to the user-allocated scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedian_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            IntPtr pBuffer);

        /// <summary>
        /// Four channel 32-bit floating-point median filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="pBuffer">Pointer to the user-allocated scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedian_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            IntPtr pBuffer);

        /// <summary>
        /// Four channel 32-bit floating-point median filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="oAnchor">X and Y offsets of the kernel origin frame of reference relative to the source pixel.</param>
        /// <param name="pBuffer">Pointer to the user-allocated scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedian_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            NppiPoint oAnchor,
            IntPtr pBuffer);

        /// <summary>
        /// Single channel 8-bit unsigned median filter scratch memory size.
        /// </summary>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="nBufferSize">Pointer to the size of the scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedianGetBufferSize_8u_C1R(
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            IntPtr nBufferSize);

        /// <summary>
        /// Three channel 8-bit unsigned median filter scratch memory size.
        /// </summary>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="nBufferSize">Pointer to the size of the scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedianGetBufferSize_8u_C3R(
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            IntPtr nBufferSize);

        /// <summary>
        /// Four channel 8-bit unsigned median filter scratch memory size.
        /// </summary>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="nBufferSize">Pointer to the size of the scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedianGetBufferSize_8u_C4R(
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            IntPtr nBufferSize);

        /// <summary>
        /// Four channel 8-bit unsigned median filter, ignoring alpha channel.
        /// </summary>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="nBufferSize">Pointer to the size of the scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedianGetBufferSize_8u_AC4R(
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            IntPtr nBufferSize);

        /// <summary>
        /// Single channel 16-bit unsigned median filter scratch memory size.
        /// </summary>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="nBufferSize">Pointer to the size of the scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedianGetBufferSize_16u_C1R(
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            IntPtr nBufferSize);

        /// <summary>
        /// Three channel 16-bit unsigned median filter scratch memory size.
        /// </summary>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="nBufferSize">Pointer to the size of the scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedianGetBufferSize_16u_C3R(
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            IntPtr nBufferSize);

        /// <summary>
        /// Four channel 16-bit unsigned median filter scratch memory size.
        /// </summary>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="nBufferSize">Pointer to the size of the scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedianGetBufferSize_16u_C4R(
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            IntPtr nBufferSize);

        /// <summary>
        /// Four channel 16-bit unsigned median filter, ignoring alpha channel.
        /// </summary>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="nBufferSize">Pointer to the size of the scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedianGetBufferSize_16u_AC4R(
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            IntPtr nBufferSize);

        /// <summary>
        /// Single channel 16-bit signed median filter scratch memory size.
        /// </summary>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="nBufferSize">Pointer to the size of the scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedianGetBufferSize_16s_C1R(
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            IntPtr nBufferSize);

        /// <summary>
        /// Three channel 16-bit signed median filter scratch memory size.
        /// </summary>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="nBufferSize">Pointer to the size of the scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedianGetBufferSize_16s_C3R(
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            IntPtr nBufferSize);

        /// <summary>
        /// Four channel 16-bit signed median filter scratch memory size.
        /// </summary>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="nBufferSize">Pointer to the size of the scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedianGetBufferSize_16s_C4R(
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            IntPtr nBufferSize);

        /// <summary>
        /// Four channel 16-bit signed median filter, ignoring alpha channel.
        /// </summary>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="nBufferSize">Pointer to the size of the scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedianGetBufferSize_16s_AC4R(
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            IntPtr nBufferSize);

        /// <summary>
        /// Single channel 32-bit floating-point median filter scratch memory size.
        /// </summary>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="nBufferSize">Pointer to the size of the scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedianGetBufferSize_32f_C1R(
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            IntPtr nBufferSize);

        /// <summary>
        /// Three channel 32-bit floating-point median filter scratch memory size.
        /// </summary>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="nBufferSize">Pointer to the size of the scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedianGetBufferSize_32f_C3R(
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            IntPtr nBufferSize);

        /// <summary>
        /// Four channel 32-bit floating-point median filter scratch memory size.
        /// </summary>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="nBufferSize">Pointer to the size of the scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedianGetBufferSize_32f_C4R(
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            IntPtr nBufferSize);

        /// <summary>
        /// Four channel 32-bit floating-point median filter, ignoring alpha channel.
        /// </summary>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="oMaskSize">Width and Height of the neighborhood region for the local Median operation.</param>
        /// <param name="nBufferSize">Pointer to the size of the scratch buffer required for the Median operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterMedianGetBufferSize_32f_AC4R(
            NppiSize oSizeROI,
            NppiSize oMaskSize,
            IntPtr nBufferSize);

        /// <summary>
        /// Single channel 8-bit unsigned horizontal Prewitt filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHoriz_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 8-bit unsigned horizontal Prewitt filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHoriz_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned horizontal Prewitt filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHoriz_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned horizontal Prewitt filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHoriz_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 16-bit signed horizontal Prewitt filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHoriz_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 16-bit signed horizontal Prewitt filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHoriz_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit signed horizontal Prewitt filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHoriz_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit signed horizontal Prewitt filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHoriz_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 32-bit floating-point horizontal Prewitt filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHoriz_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 32-bit floating-point horizontal Prewitt filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHoriz_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 32-bit floating-point horizontal Prewitt filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHoriz_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 32-bit floating-point horizontal Prewitt filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHoriz_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit unsigned horizontal Prewitt filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHorizBorder_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned horizontal Prewitt filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHorizBorder_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned horizontal Prewitt filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHorizBorder_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned horizontal Prewitt filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHorizBorder_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit signed horizontal Prewitt filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHorizBorder_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit signed horizontal Prewitt filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHorizBorder_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed horizontal Prewitt filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHorizBorder_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed horizontal Prewitt filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHorizBorder_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit floating-point horizontal Prewitt filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHorizBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 32-bit floating-point horizontal Prewitt filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHorizBorder_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point horizontal Prewitt filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHorizBorder_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point horizontal Prewitt filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittHorizBorder_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned vertical Prewitt filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVert_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 8-bit unsigned vertical Prewitt filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVert_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned vertical Prewitt filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVert_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned vertical Prewitt filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVert_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 16-bit signed vertical Prewitt filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVert_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 16-bit signed vertical Prewitt filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVert_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit signed vertical Prewitt filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVert_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit signed vertical Prewitt filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVert_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 32-bit floating-point vertical Prewitt filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVert_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 32-bit floating-point vertical Prewitt filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVert_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 32-bit floating-point vertical Prewitt filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVert_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 32-bit floating-point vertical Prewitt filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVert_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit unsigned vertical Prewitt filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVertBorder_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned vertical Prewitt filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVertBorder_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned vertical Prewitt filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVertBorder_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned vertical Prewitt filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVertBorder_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit signed vertical Prewitt filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVertBorder_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit signed vertical Prewitt filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVertBorder_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed vertical Prewitt filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVertBorder_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed vertical Prewitt filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVertBorder_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit floating-point vertical Prewitt filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVertBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 32-bit floating-point vertical Prewitt filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVertBorder_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point vertical Prewitt filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVertBorder_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point vertical Prewitt filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterPrewittVertBorder_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned to 16-bit signed horizontal Scharr filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterScharrHoriz_8u16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit signed to 16-bit signed horizontal Scharr filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterScharrHoriz_8s16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 32-bit floating-point horizontal Scharr filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterScharrHoriz_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit unsigned to 16-bit signed vertical Scharr filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterScharrVert_8u16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit signed to 16-bit signed vertical Scharr filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterScharrVert_8s16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 32-bit floating-point vertical Scharr filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterScharrVert_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit unsigned to 16-bit signed horizontal Scharr filter kernel with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterScharrHorizBorder_8u16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit signed to 16-bit signed horizontal Scharr filter kernel with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterScharrHorizBorder_8s16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit floating-point horizontal Scharr filter kernel with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterScharrHorizBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned to 16-bit signed vertical Scharr filter kernel with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterScharrVertBorder_8u16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit signed to 16-bit signed vertical Scharr filter kernel with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterScharrVertBorder_8s16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit floating-point vertical Scharr filter kernel with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterScharrVertBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned horizontal Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHoriz_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 8-bit unsigned horizontal Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHoriz_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned horizontal Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHoriz_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit signed horizontal Sobel filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHoriz_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 16-bit signed horizontal Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHoriz_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 16-bit signed horizontal Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHoriz_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit signed horizontal Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHoriz_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned horizontal Sobel filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHoriz_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 32-bit floating-point horizontal Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHoriz_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 32-bit floating-point horizontal Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHoriz_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 32-bit floating-point horizontal Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHoriz_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 32-bit floating-point horizontal Sobel filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHoriz_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit unsigned to 16-bit signed horizontal Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHoriz_8u16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 8-bit signed to 16-bit signed horizontal Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHoriz_8s16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 32-bit floating-point horizontal Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHorizMask_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 8-bit unsigned vertical Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVert_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 8-bit unsigned vertical Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVert_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned vertical Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVert_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit signed vertical Sobel filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVert_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 16-bit signed vertical Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVert_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 16-bit signed vertical Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVert_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit signed vertical Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVert_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned vertical Sobel filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVert_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 32-bit floating-point vertical Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVert_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 32-bit floating-point vertical Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVert_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 32-bit floating-point vertical Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVert_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 32-bit floating-point vertical Sobel filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVert_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit unsigned to 16-bit signed vertical Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVert_8u16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 8-bit signed to 16-bit signed vertical Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVert_8s16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 32-bit floating-point vertical Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVertMask_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 8-bit unsigned to 16-bit signed second derivative, horizontal Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHorizSecond_8u16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 8-bit signed to 16-bit signed second derivative, horizontal Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHorizSecond_8s16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 32-bit floating-point second derivative, horizontal Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHorizSecond_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 8-bit unsigned to 16-bit signed second derivative, vertical Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVertSecond_8u16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 8-bit signed to 16-bit signed second derivative, vertical Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVertSecond_8s16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 32-bit floating-point second derivative, vertical Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVertSecond_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 8-bit unsigned to 16-bit signed second cross derivative Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelCross_8u16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 8-bit signed to 16-bit signed second cross derivative Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelCross_8s16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 32-bit floating-point second cross derivative Sobel filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelCross_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 8-bit unsigned horizontal Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHorizBorder_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned horizontal Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHorizBorder_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned horizontal Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHorizBorder_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed horizontal Sobel filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHorizBorder_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit signed horizontal Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHorizBorder_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit signed horizontal Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHorizBorder_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed horizontal Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHorizBorder_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned horizontal Sobel filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHorizBorder_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit floating-point horizontal Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHorizBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 32-bit floating-point horizontal Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHorizBorder_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point horizontal Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHorizBorder_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point horizontal Sobel filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHorizBorder_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned to 16-bit signed horizontal Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHorizBorder_8u16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit signed to 16-bit signed horizontal Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHorizBorder_8s16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit floating-point horizontal Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHorizMaskBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned vertical Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVertBorder_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned vertical Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVertBorder_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned vertical Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVertBorder_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed vertical Sobel filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVertBorder_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit signed vertical Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVertBorder_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit signed vertical Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVertBorder_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed vertical Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVertBorder_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned vertical Sobel filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVertBorder_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit floating-point vertical Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVertBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 32-bit floating-point vertical Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVertBorder_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point vertical Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVertBorder_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point vertical Sobel filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVertBorder_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned to 16-bit signed vertical Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVertBorder_8u16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit signed to 16-bit signed vertical Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVertBorder_8s16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit floating-point vertical Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVertMaskBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned to 16-bit signed second derivative, horizontal Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHorizSecondBorder_8u16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit signed to 16-bit signed second derivative, horizontal Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHorizSecondBorder_8s16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit floating-point second derivative, horizontal Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelHorizSecondBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned to 16-bit signed second derivative, vertical Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVertSecondBorder_8u16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit signed to 16-bit signed second derivative, vertical Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVertSecondBorder_8s16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit floating-point second derivative, vertical Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelVertSecondBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned to 16-bit signed second cross derivative Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelCrossBorder_8u16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit signed to 16-bit signed second cross derivative Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelCrossBorder_8s16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit floating-point second cross derivative Sobel filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSobelCrossBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned horizontal Roberts filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDown_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 8-bit unsigned horizontal Roberts filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDown_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned horizontal Roberts filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDown_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned horizontal Roberts filter, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDown_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 16-bit signed horizontal Roberts filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDown_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 16-bit signed horizontal Roberts filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDown_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit signed horizontal Roberts filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDown_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit signed horizontal Roberts filter, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDown_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 32-bit floating-point horizontal Roberts filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDown_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 32-bit floating-point horizontal Roberts filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDown_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 32-bit floating-point horizontal Roberts filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDown_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 32-bit floating-point horizontal Roberts filter, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDown_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit unsigned horizontal Roberts filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDownBorder_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned horizontal Roberts filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDownBorder_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned horizontal Roberts filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDownBorder_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned horizontal Roberts filter with border control, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDownBorder_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit signed horizontal Roberts filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDownBorder_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit signed horizontal Roberts filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDownBorder_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed horizontal Roberts filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDownBorder_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed horizontal Roberts filter with border control, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDownBorder_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit floating-point horizontal Roberts filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDownBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 32-bit floating-point horizontal Roberts filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDownBorder_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point horizontal Roberts filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDownBorder_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point horizontal Roberts filter with border control, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsDownBorder_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned vertical Roberts filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUp_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 8-bit unsigned vertical Roberts filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUp_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned vertical Roberts filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUp_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned vertical Roberts filter, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUp_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 16-bit signed vertical Roberts filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUp_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 16-bit signed vertical Roberts filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUp_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit signed vertical Roberts filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUp_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit signed vertical Roberts filter, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUp_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 32-bit floating-point vertical Roberts filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUp_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 32-bit floating-point vertical Roberts filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUp_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 32-bit floating-point vertical Roberts filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUp_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 32-bit floating-point vertical Roberts filter, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUp_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit unsigned vertical Roberts filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUpBorder_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned vertical Roberts filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUpBorder_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned vertical Roberts filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUpBorder_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned vertical Roberts filter with border control, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUpBorder_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit signed vertical Roberts filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUpBorder_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit signed vertical Roberts filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUpBorder_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed vertical Roberts filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUpBorder_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed vertical Roberts filter with border control, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUpBorder_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit floating-point vertical Roberts filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUpBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 32-bit floating-point vertical Roberts filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUpBorder_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point vertical Roberts filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUpBorder_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point vertical Roberts filter with border control, ignoring alpha-channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterRobertsUpBorder_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned Laplace filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplace_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Three channel 8-bit unsigned Laplace filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplace_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 8-bit unsigned Laplace filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplace_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 8-bit unsigned Laplace filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplace_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 16-bit signed Laplace filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplace_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Three channel 16-bit signed Laplace filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplace_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 16-bit signed Laplace filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplace_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 16-bit signed Laplace filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplace_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 32-bit floating-point Laplace filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplace_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Three channel 32-bit floating-point Laplace filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplace_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 32-bit floating-point Laplace filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplace_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 32-bit floating-point Laplace filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplace_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 8-bit unsigned to 16-bit signed Laplace filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplace_8u16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 8-bit signed to 16-bit signed Laplace filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplace_8s16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 8-bit unsigned Laplace filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplaceBorder_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned Laplace filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplaceBorder_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned Laplace filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplaceBorder_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned Laplace filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplaceBorder_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit signed Laplace filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplaceBorder_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit signed Laplace filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplaceBorder_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed Laplace filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplaceBorder_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed Laplace filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplaceBorder_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit floating-point Laplace filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplaceBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 32-bit floating-point Laplace filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplaceBorder_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point Laplace filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplaceBorder_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point Laplace filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplaceBorder_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned to 16-bit signed Laplace filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplaceBorder_8u16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit signed to 16-bit signed Laplace filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLaplaceBorder_8s16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGauss_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Three channel 8-bit unsigned Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGauss_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 8-bit unsigned Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGauss_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 8-bit unsigned Gauss filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGauss_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 16-bit unsigned Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGauss_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Three channel 16-bit unsigned Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGauss_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 16-bit unsigned Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGauss_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 16-bit unsigned Gauss filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGauss_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 16-bit signed Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGauss_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Three channel 16-bit signed Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGauss_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 16-bit signed Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGauss_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 16-bit signed Gauss filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGauss_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 32-bit floating-point Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGauss_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Three channel 32-bit floating-point Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGauss_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 32-bit floating-point Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGauss_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 32-bit floating-point Gauss filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGauss_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 8-bit unsigned Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvanced_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel);

        /// <summary>
        /// Three channel 8-bit unsigned Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvanced_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel);

        /// <summary>
        /// Four channel 8-bit unsigned Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvanced_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel);

        /// <summary>
        /// Four channel 8-bit unsigned Gauss filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvanced_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel);

        /// <summary>
        /// Single channel 16-bit unsigned Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvanced_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel);

        /// <summary>
        /// Three channel 16-bit unsigned Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvanced_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel);

        /// <summary>
        /// Four channel 16-bit unsigned Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvanced_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel);

        /// <summary>
        /// Four channel 16-bit unsigned Gauss filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvanced_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel);

        /// <summary>
        /// Single channel 16-bit signed Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvanced_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel);

        /// <summary>
        /// Three channel 16-bit signed Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvanced_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel);

        /// <summary>
        /// Four channel 16-bit signed Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvanced_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel);

        /// <summary>
        /// Four channel 16-bit signed Gauss filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvanced_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel);

        /// <summary>
        /// Single channel 32-bit floating-point Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvanced_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel);

        /// <summary>
        /// Three channel 32-bit floating-point Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvanced_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel);

        /// <summary>
        /// Four channel 32-bit floating-point Gauss filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvanced_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel);

        /// <summary>
        /// Four channel 32-bit floating-point Gauss filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvanced_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel);

        /// <summary>
        /// Single channel 8-bit unsigned Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussBorder_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussBorder_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussBorder_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned Gauss filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussBorder_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit unsigned Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussBorder_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit unsigned Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussBorder_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussBorder_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned Gauss filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussBorder_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit signed Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussBorder_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit signed Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussBorder_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussBorder_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed Gauss filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussBorder_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit floating-point Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 32-bit floating-point Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussBorder_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussBorder_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point Gauss filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussBorder_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvancedBorder_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvancedBorder_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvancedBorder_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned Gauss filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvancedBorder_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit unsigned Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvancedBorder_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit unsigned Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvancedBorder_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvancedBorder_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned Gauss filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvancedBorder_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit signed Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvancedBorder_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit signed Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvancedBorder_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvancedBorder_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed Gauss filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvancedBorder_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit floating-point Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvancedBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 32-bit floating-point Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvancedBorder_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point Gauss filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvancedBorder_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point Gauss filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nFilterTaps">The number of filter taps where nFilterTaps = 2 * ((int)((float)ceil(radius) + 0.5F) ) + 1.</param>
        /// <param name="pKernel">Pointer to an array of nFilterTaps kernel coefficients which sum to 1.0F.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterGaussAdvancedBorder_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            int nFilterTaps,
            IntPtr pKernel,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPass_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Three channel 8-bit unsigned high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPass_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 8-bit unsigned high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPass_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 8-bit unsigned high-pass filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPass_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 16-bit unsigned high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPass_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Three channel 16-bit unsigned high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPass_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 16-bit unsigned high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPass_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 16-bit unsigned high-pass filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPass_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 16-bit signed high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPass_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Three channel 16-bit signed high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPass_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 16-bit signed high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPass_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 16-bit signed high-pass filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPass_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 32-bit floating-point high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPass_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Three channel 32-bit floating-point high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPass_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 32-bit floating-point high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPass_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 32-bit floating-point high-pass filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPass_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 8-bit unsigned high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPassBorder_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPassBorder_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPassBorder_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned high-pass filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPassBorder_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit unsigned high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPassBorder_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit unsigned high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPassBorder_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPassBorder_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned high-pass filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPassBorder_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit signed high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPassBorder_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit signed high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPassBorder_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPassBorder_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed high-pass filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPassBorder_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit floating-point high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPassBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 32-bit floating-point high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPassBorder_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPassBorder_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point high-pass filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterHighPassBorder_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned low-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPass_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Three channel 8-bit unsigned low-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPass_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 8-bit unsigned low-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPass_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 8-bit unsigned low-pass filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPass_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 16-bit unsigned low-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPass_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Three channel 16-bit unsigned low-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPass_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 16-bit unsigned low-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPass_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 16-bit unsigned low-pass filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPass_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 16-bit signed low-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPass_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Three channel 16-bit signed low-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPass_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 16-bit signed low-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPass_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 16-bit signed low-pass filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPass_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 32-bit floating-point low-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPass_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Three channel 32-bit floating-point low-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPass_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 32-bit floating-point low-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPass_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Four channel 32-bit floating-point high-pass filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPass_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize);

        /// <summary>
        /// Single channel 8-bit unsigned high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPassBorder_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPassBorder_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPassBorder_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned high-pass filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPassBorder_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit unsigned high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPassBorder_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit unsigned high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPassBorder_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPassBorder_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned high-pass filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPassBorder_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit signed high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPassBorder_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit signed high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPassBorder_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPassBorder_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed high-pass filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPassBorder_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit floating-point high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPassBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 32-bit floating-point high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPassBorder_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point high-pass filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPassBorder_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point high-pass filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eMaskSize">Enumeration value specifying the mask size.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterLowPassBorder_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiMaskSize eMaskSize,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned sharpening filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpen_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 8-bit unsigned sharpening filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpen_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned sharpening filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpen_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 8-bit unsigned sharpening filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpen_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 16-bit unsigned sharpening filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpen_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 16-bit unsigned sharpening filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpen_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit unsigned sharpening filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpen_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit unsigned sharpening filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpen_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 16-bit signed sharpening filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpen_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 16-bit signed sharpening filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpen_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit signed sharpening filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpen_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 16-bit signed sharpening filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpen_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 32-bit floating-point sharpening filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpen_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Three channel 32-bit floating-point sharpening filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpen_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 32-bit floating-point sharpening filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpen_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Four channel 32-bit floating-point sharpening filter, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpen_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI);

        /// <summary>
        /// Single channel 8-bit unsigned sharpening filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpenBorder_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 8-bit unsigned sharpening filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpenBorder_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned sharpening filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpenBorder_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 8-bit unsigned sharpening filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpenBorder_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit unsigned sharpening filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpenBorder_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit unsigned sharpening filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpenBorder_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned sharpening filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpenBorder_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit unsigned sharpening filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpenBorder_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 16-bit signed sharpening filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpenBorder_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 16-bit signed sharpening filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpenBorder_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed sharpening filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpenBorder_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 16-bit signed sharpening filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpenBorder_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 32-bit floating-point sharpening filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpenBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Three channel 32-bit floating-point sharpening filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpenBorder_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point sharpening filter with border control.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpenBorder_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Four channel 32-bit floating-point sharpening filter with border control, ignoring alpha channel.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcSize">Source image width and height in pixels relative to pSrc.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterSharpenBorder_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiSize oSrcSize,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            NppiBorderType eBorderType);

        /// <summary>
        /// Single channel 8-bit unsigned unsharp filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="nWeight">The percentage of the difference between the original and the high pass image that is added back into the original.</param>
        /// <param name="nThreshold">The threshold neede to apply the difference amount.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <param name="pDeviceBuffer">Pointer to the user-allocated device scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpBorder_8u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            Npp32f nRadius,
            Npp32f nSigma,
            Npp32f nWeight,
            Npp32f nThreshold,
            NppiBorderType eBorderType,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Three channel 8-bit unsigned unsharp filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="nWeight">The percentage of the difference between the original and the high pass image that is added back into the original.</param>
        /// <param name="nThreshold">The threshold neede to apply the difference amount.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <param name="pDeviceBuffer">Pointer to the user-allocated device scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpBorder_8u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            Npp32f nRadius,
            Npp32f nSigma,
            Npp32f nWeight,
            Npp32f nThreshold,
            NppiBorderType eBorderType,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Four channel 8-bit unsigned unsharp filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="nWeight">The percentage of the difference between the original and the high pass image that is added back into the original.</param>
        /// <param name="nThreshold">The threshold neede to apply the difference amount.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <param name="pDeviceBuffer">Pointer to the user-allocated device scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpBorder_8u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            Npp32f nRadius,
            Npp32f nSigma,
            Npp32f nWeight,
            Npp32f nThreshold,
            NppiBorderType eBorderType,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Four channel 8-bit unsigned unsharp filter (alpha channel is not processed).
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="nWeight">The percentage of the difference between the original and the high pass image that is added back into the original.</param>
        /// <param name="nThreshold">The threshold neede to apply the difference amount.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <param name="pDeviceBuffer">Pointer to the user-allocated device scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpBorder_8u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            Npp32f nRadius,
            Npp32f nSigma,
            Npp32f nWeight,
            Npp32f nThreshold,
            NppiBorderType eBorderType,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Single channel 16-bit unsigned unsharp filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="nWeight">The percentage of the difference between the original and the high pass image that is added back into the original.</param>
        /// <param name="nThreshold">The threshold neede to apply the difference amount.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <param name="pDeviceBuffer">Pointer to the user-allocated device scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpBorder_16u_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            Npp32f nRadius,
            Npp32f nSigma,
            Npp32f nWeight,
            Npp32f nThreshold,
            NppiBorderType eBorderType,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Three channel 16-bit unsigned unsharp filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="nWeight">The percentage of the difference between the original and the high pass image that is added back into the original.</param>
        /// <param name="nThreshold">The threshold neede to apply the difference amount.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <param name="pDeviceBuffer">Pointer to the user-allocated device scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpBorder_16u_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            Npp32f nRadius,
            Npp32f nSigma,
            Npp32f nWeight,
            Npp32f nThreshold,
            NppiBorderType eBorderType,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Four channel 16-bit unsigned unsharp filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="nWeight">The percentage of the difference between the original and the high pass image that is added back into the original.</param>
        /// <param name="nThreshold">The threshold neede to apply the difference amount.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <param name="pDeviceBuffer">Pointer to the user-allocated device scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpBorder_16u_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            Npp32f nRadius,
            Npp32f nSigma,
            Npp32f nWeight,
            Npp32f nThreshold,
            NppiBorderType eBorderType,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Four channel 16-bit unsigned unsharp filter (alpha channel is not processed).
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="nWeight">The percentage of the difference between the original and the high pass image that is added back into the original.</param>
        /// <param name="nThreshold">The threshold neede to apply the difference amount.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <param name="pDeviceBuffer">Pointer to the user-allocated device scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpBorder_16u_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            Npp32f nRadius,
            Npp32f nSigma,
            Npp32f nWeight,
            Npp32f nThreshold,
            NppiBorderType eBorderType,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Single channel 16-bit signed unsharp filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="nWeight">The percentage of the difference between the original and the high pass image that is added back into the original.</param>
        /// <param name="nThreshold">The threshold neede to apply the difference amount.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <param name="pDeviceBuffer">Pointer to the user-allocated device scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpBorder_16s_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            Npp32f nRadius,
            Npp32f nSigma,
            Npp32f nWeight,
            Npp32f nThreshold,
            NppiBorderType eBorderType,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Three channel 16-bit signed unsharp filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="nWeight">The percentage of the difference between the original and the high pass image that is added back into the original.</param>
        /// <param name="nThreshold">The threshold neede to apply the difference amount.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <param name="pDeviceBuffer">Pointer to the user-allocated device scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpBorder_16s_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            Npp32f nRadius,
            Npp32f nSigma,
            Npp32f nWeight,
            Npp32f nThreshold,
            NppiBorderType eBorderType,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Four channel 16-bit signed unsharp filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="nWeight">The percentage of the difference between the original and the high pass image that is added back into the original.</param>
        /// <param name="nThreshold">The threshold neede to apply the difference amount.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <param name="pDeviceBuffer">Pointer to the user-allocated device scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpBorder_16s_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            Npp32f nRadius,
            Npp32f nSigma,
            Npp32f nWeight,
            Npp32f nThreshold,
            NppiBorderType eBorderType,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Four channel 16-bit signed unsharp filter (alpha channel is not processed).
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="nWeight">The percentage of the difference between the original and the high pass image that is added back into the original.</param>
        /// <param name="nThreshold">The threshold neede to apply the difference amount.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <param name="pDeviceBuffer">Pointer to the user-allocated device scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpBorder_16s_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            Npp32f nRadius,
            Npp32f nSigma,
            Npp32f nWeight,
            Npp32f nThreshold,
            NppiBorderType eBorderType,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Single channel 32-bit floating point unsharp filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="nWeight">The percentage of the difference between the original and the high pass image that is added back into the original.</param>
        /// <param name="nThreshold">The threshold neede to apply the difference amount.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <param name="pDeviceBuffer">Pointer to the user-allocated device scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpBorder_32f_C1R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            Npp32f nRadius,
            Npp32f nSigma,
            Npp32f nWeight,
            Npp32f nThreshold,
            NppiBorderType eBorderType,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Three channel 32-bit floating point unsharp filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="nWeight">The percentage of the difference between the original and the high pass image that is added back into the original.</param>
        /// <param name="nThreshold">The threshold neede to apply the difference amount.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <param name="pDeviceBuffer">Pointer to the user-allocated device scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpBorder_32f_C3R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            Npp32f nRadius,
            Npp32f nSigma,
            Npp32f nWeight,
            Npp32f nThreshold,
            NppiBorderType eBorderType,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Four channel 32-bit floating point unsharp filter.
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="nWeight">The percentage of the difference between the original and the high pass image that is added back into the original.</param>
        /// <param name="nThreshold">The threshold neede to apply the difference amount.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <param name="pDeviceBuffer">Pointer to the user-allocated device scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpBorder_32f_C4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            Npp32f nRadius,
            Npp32f nSigma,
            Npp32f nWeight,
            Npp32f nThreshold,
            NppiBorderType eBorderType,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Four channel 32-bit floating point unsharp filter (alpha channel is not processed).
        /// </summary>
        /// <param name="pSrc">source_image_pointer.</param>
        /// <param name="nSrcStep">source_image_line_step.</param>
        /// <param name="oSrcOffset">The pixel offset that pSrc points to relative to the origin of the source image.</param>
        /// <param name="pDst">destination_image_pointer.</param>
        /// <param name="nDstStep">destination_image_line_step.</param>
        /// <param name="oSizeROI">roi_specification.</param>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="nWeight">The percentage of the difference between the original and the high pass image that is added back into the original.</param>
        /// <param name="nThreshold">The threshold neede to apply the difference amount.</param>
        /// <param name="eBorderType">The border type operation to be applied at source image border boundaries.</param>
        /// <param name="pDeviceBuffer">Pointer to the user-allocated device scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes, roi_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpBorder_32f_AC4R(
            IntPtr pSrc,
            Npp32s nSrcStep,
            NppiPoint oSrcOffset,
            IntPtr pDst,
            Npp32s nDstStep,
            NppiSize oSizeROI,
            Npp32f nRadius,
            Npp32f nSigma,
            Npp32f nWeight,
            Npp32f nThreshold,
            NppiBorderType eBorderType,
            IntPtr pDeviceBuffer);

        /// <summary>
        /// Single channel 8-bit unsigned unsharp filter scratch memory size.
        /// </summary>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="hpBufferSize">Pointer to the size of the scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpGetBufferSize_8u_C1R(
            Npp32f nRadius,
            Npp32f nSigma,
            IntPtr hpBufferSize);

        /// <summary>
        /// Three channel 8-bit unsigned unsharp filter scratch memory size.
        /// </summary>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="hpBufferSize">Pointer to the size of the scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpGetBufferSize_8u_C3R(
            Npp32f nRadius,
            Npp32f nSigma,
            IntPtr hpBufferSize);

        /// <summary>
        /// Four channel 8-bit unsigned unsharp filter scratch memory size.
        /// </summary>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="hpBufferSize">Pointer to the size of the scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpGetBufferSize_8u_C4R(
            Npp32f nRadius,
            Npp32f nSigma,
            IntPtr hpBufferSize);

        /// <summary>
        /// Four channel 8-bit unsigned unsharp filter scratch memory size (alpha channel is not processed).
        /// </summary>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="hpBufferSize">Pointer to the size of the scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpGetBufferSize_8u_AC4R(
            Npp32f nRadius,
            Npp32f nSigma,
            IntPtr hpBufferSize);

        /// <summary>
        /// Single channel 16-bit unsigned unsharp filter scratch memory size.
        /// </summary>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="hpBufferSize">Pointer to the size of the scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpGetBufferSize_16u_C1R(
            Npp32f nRadius,
            Npp32f nSigma,
            IntPtr hpBufferSize);

        /// <summary>
        /// Three channel 16-bit unsigned unsharp filter scratch memory size.
        /// </summary>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="hpBufferSize">Pointer to the size of the scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpGetBufferSize_16u_C3R(
            Npp32f nRadius,
            Npp32f nSigma,
            IntPtr hpBufferSize);

        /// <summary>
        /// Four channel 16-bit unsigned unsharp filter scratch memory size.
        /// </summary>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="hpBufferSize">Pointer to the size of the scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpGetBufferSize_16u_C4R(
            Npp32f nRadius,
            Npp32f nSigma,
            IntPtr hpBufferSize);

        /// <summary>
        /// Four channel 16-bit unsigned unsharp filter scratch memory size (alpha channel is not processed).
        /// </summary>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="hpBufferSize">Pointer to the size of the scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpGetBufferSize_16u_AC4R(
            Npp32f nRadius,
            Npp32f nSigma,
            IntPtr hpBufferSize);

        /// <summary>
        /// Single channel 16-bit signed unsharp filter scratch memory size.
        /// </summary>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="hpBufferSize">Pointer to the size of the scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpGetBufferSize_16s_C1R(
            Npp32f nRadius,
            Npp32f nSigma,
            IntPtr hpBufferSize);

        /// <summary>
        /// Three channel 16-bit signed unsharp filter scratch memory size.
        /// </summary>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="hpBufferSize">Pointer to the size of the scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpGetBufferSize_16s_C3R(
            Npp32f nRadius,
            Npp32f nSigma,
            IntPtr hpBufferSize);

        /// <summary>
        /// Four channel 16-bit signed unsharp filter scratch memory size.
        /// </summary>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="hpBufferSize">Pointer to the size of the scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpGetBufferSize_16s_C4R(
            Npp32f nRadius,
            Npp32f nSigma,
            IntPtr hpBufferSize);

        /// <summary>
        /// Four channel 16-bit signed unsharp filter scratch memory size (alpha channel is not processed).
        /// </summary>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="hpBufferSize">Pointer to the size of the scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpGetBufferSize_16s_AC4R(
            Npp32f nRadius,
            Npp32f nSigma,
            IntPtr hpBufferSize);

        /// <summary>
        /// Single channel 32-bit floating point unsharp filter scratch memory size.
        /// </summary>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="hpBufferSize">Pointer to the size of the scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpGetBufferSize_32f_C1R(
            Npp32f nRadius,
            Npp32f nSigma,
            IntPtr hpBufferSize);

        /// <summary>
        /// Three channel 32-bit floating point unsharp filter scratch memory size.
        /// </summary>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="hpBufferSize">Pointer to the size of the scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpGetBufferSize_32f_C3R(
            Npp32f nRadius,
            Npp32f nSigma,
            IntPtr hpBufferSize);

        /// <summary>
        /// Four channel 32-bit floating point unsharp filter scratch memory size.
        /// </summary>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="hpBufferSize">Pointer to the size of the scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpGetBufferSize_32f_C4R(
            Npp32f nRadius,
            Npp32f nSigma,
            IntPtr hpBufferSize);

        /// <summary>
        /// Four channel 32-bit floating point unsharp filter scratch memory size (alpha channel is not processed).
        /// </summary>
        /// <param name="nRadius">The radius of the Gaussian filter, in pixles, not counting the center pixel.</param>
        /// <param name="nSigma">The standard deviation of the Gaussian filter, in pixel.</param>
        /// <param name="hpBufferSize">Pointer to the size of the scratch buffer required for the unsharp operation.</param>
        /// <returns>image_data_error_codes</returns>
        [DllImport(fDll, SetLastError = true)]
        public static extern NppStatus nppiFilterUnsharpGetBufferSize_32f_AC4R(
            Npp32f nRadius,
            Npp32f nSigma,
            IntPtr hpBufferSize);

    }
}